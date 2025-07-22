"""
This module defines the FastAPI application for the Structure from Motion (SfM) pipeline.
It includes endpoints for initiating SfM processing, checking job status, and monitoring compute usage.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
import os
import uuid
import asyncio
import psutil # For compute_status endpoint
from datetime import datetime
import json # For saving/loading JSON outputs

# Import the actual SfMProcessor from models.py
from models import SfMProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Structure from Motion (SfM) API",
    description="API for processing videos to estimate camera parameters and generate 3D point clouds using VGGT.",
    version="1.0.0"
)

# Configuration directories (relative to /app in Docker)
VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/outputs"
VGGT_REPO_PATH = "/app/vggt" # Path where VGGT repository is cloned in Dockerfile

# In-memory storage for job statuses
# NOTE: This will not persist across API restarts. For production, use a database.
job_statuses = {}

# Global SfMProcessor instance
# This will be initialized once when the application starts
sfm_processor_instance: SfMProcessor = None

# Ensure necessary data directories exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs when the FastAPI application starts up.
    Used to initialize the global SfMProcessor instance (and thus load the VGGT model).
    """
    global sfm_processor_instance
    print(f"[{datetime.now()}] APP STARTUP: Initializing SfMProcessor and loading VGGT model...")
    try:
        # Initialize SfMProcessor. This will load the VGGT model.
        # This potentially long-running operation is done once at startup.
        sfm_processor_instance = SfMProcessor(vggt_repo_path=VGGT_REPO_PATH)
        print(f"[{datetime.now()}] APP STARTUP: SfMProcessor and VGGT model loaded successfully.")
    except Exception as e:
        print(f"[{datetime.now()}] APP STARTUP ERROR: Failed to load SfMProcessor/VGGT model: {e}")
        # Depending on criticality, you might want to raise an exception here
        # to prevent the app from starting if the model is essential.
        # For now, we'll let it start but log the error.

@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    print(f"[{datetime.now()}] API Request: / received.")
    return {"message": "Welcome to the SfM Pipeline API."}

def update_sfm_progress_callback(job_id: str, progress_percent: int, stage: str):
    """
    Callback function to update the SfM job status with real-time progress.
    """
    job_statuses[job_id].update({
        "progress": f"{progress_percent}%",
        "stage": stage,
        "message": f"SfM processing: {stage} ({progress_percent}%)"
    })
    print(f"[{datetime.now()}] Job {job_id} progress: {progress_percent}% - {stage}")


async def process_sfm_background(job_id: str, input_video_path: str, output_base_path: str, sample_frames: int):
    """
    Background task to perform Structure from Motion processing.
    """
    global sfm_processor_instance
    if sfm_processor_instance is None:
        job_statuses[job_id].update({
            "status": "FAILED",
            "message": "SfMProcessor not initialized. Model loading failed at startup.",
            "error": "Model not ready.",
            "progress": "Error",
            "stage": "Initialization Failed"
        })
        print(f"[{datetime.now()}] Job {job_id}: SfMProcessor not ready, cannot process.")
        return

    job_statuses[job_id] = {"status": "PROCESSING", "message": "SfM job initialized.", "progress": "0%", "stage": "Starting"}
    print(f"[{datetime.now()}] Job {job_id}: Starting SfM processing for {input_video_path}")

    try:
        # Use the globally initialized SfMProcessor instance
        output_files = await sfm_processor_instance.process_video(
            video_path=input_video_path,
            output_base_path=output_base_path,
            progress_callback=lambda p, s: update_sfm_progress_callback(job_id, p, s),
            sample_frames=sample_frames # Pass the sample_frames parameter
        )

        job_statuses[job_id].update({
            "status": "COMPLETED",
            "message": "SfM processing completed successfully.",
            "output_files": output_files,
            "progress": "100%",
            "stage": "Completed"
        })
        print(f"[{datetime.now()}] Job {job_id}: SfM processing completed. Outputs: {output_files}")

    except Exception as e:
        job_statuses[job_id].update({
            "status": "FAILED",
            "message": "An error occurred during SfM processing.",
            "error": str(e),
            "progress": "Error",
            "stage": "Failed"
        })
        print(f"[{datetime.now()}] Job {job_id}: SfM processing failed: {e}")
    finally:
        # Clean up the input video file after processing (success or failure)
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
            print(f"[{datetime.now()}] Job {job_id}: Cleaned up input file {input_video_path}")


@app.post("/sfm_predict")
async def sfm_predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_name: str = Form(None, description="Optional: Base name for the output files. Defaults to a timestamped name."),
    sample_frames: int = Form(50, description="Number of frames to sample from the video. Defaults to 50.")
):
    """
    Initiates an asynchronous job to perform Structure from Motion on an uploaded video.
    """
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Use provided output_name or generate a timestamped one
    final_output_name = output_name if output_name else f"sfm_run_{timestamp}"
    
    input_video_path = os.path.join(VIDEO_DIR, f"{final_output_name}_{file.filename}")
    output_base_path = os.path.join(OUTPUT_DIR, final_output_name)

    print(f"[{datetime.now()}] API Request: /sfm_predict received for job {job_id}")

    # Check if the SfMProcessor is ready before accepting the task
    if sfm_processor_instance is None:
        print(f"[{datetime.now()}] API Request: /sfm_predict rejected - SfMProcessor not initialized.")
        raise HTTPException(status_code=503, detail="Service Unavailable: VGGT model is still loading or failed to load at startup.")

    try:
        # Save the uploaded video file
        print(f"[{datetime.now()}] Job {job_id}: Saving uploaded video to {input_video_path}")
        with open(input_video_path, "wb") as buffer:
            buffer.write(await file.read())
        print(f"[{datetime.now()}] Job {job_id}: Video saved.")

        # Initialize job status
        job_statuses[job_id] = {"status": "PENDING", "message": "SfM job received and queued.", "progress": "0%", "stage": "Queued"}

        # Add the SfM processing to background tasks
        background_tasks.add_task(process_sfm_background, job_id, input_video_path, output_base_path, sample_frames)
        print(f"[{datetime.now()}] Job {job_id}: Background task added.")

        return JSONResponse(
            status_code=202, # 202 Accepted
            content={
                "status": "accepted",
                "message": "SfM processing job started. Check status using the job ID.",
                "job_id": job_id,
                "output_name": final_output_name
            }
        )
    except Exception as e:
        # Clean up the uploaded file if an error occurs during initial file save
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
        # Remove job from status if it failed before even starting background task
        if job_id in job_statuses:
            del job_statuses[job_id]
        print(f"[{datetime.now()}] API Request: /sfm_predict failed for job {job_id}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate SfM processing: {str(e)}")


@app.get("/sfm_status/{job_id}")
async def sfm_status(job_id: str):
    """
    Checks the current status of an SfM processing job.
    """
    print(f"[{datetime.now()}] API Request: /sfm_status/{job_id} received.")
    status_info = job_statuses.get(job_id)
    if status_info:
        response_content = status_info
        headers = {}
        if status_info["status"] in ["PENDING", "PROCESSING"]:
            # Suggest polling every 5 seconds while processing
            headers["Retry-After"] = "5"
        print(f"[{datetime.now()}] API Request: /sfm_status/{job_id} returning status: {status_info['status']}")
        return JSONResponse(status_code=200, content=response_content, headers=headers)
    else:
        print(f"[{datetime.now()}] API Request: /sfm_status/{job_id} - Job ID not found.")
        raise HTTPException(status_code=404, detail="Job ID not found.")


@app.get("/compute_status")
async def compute_status():
    """
    Provides real-time information about the Docker container's CPU, memory, disk I/O, and network I/O usage.
    """
    # Use interval=None for instantaneous (non-blocking) CPU percentage
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()

    print(f"[{datetime.now()}] API Request: /compute_status received. Returning system metrics.")

    return JSONResponse(
        status_code=200,
        content={
            "cpu_percent": cpu_percent,
            "memory_percent": memory_info.percent,
            "memory_used_mb": round(memory_info.used / (1024 * 1024), 2),
            "memory_total_mb": round(memory_info.total / (1024 * 1024), 2),
            "disk_read_bytes": disk_io.read_bytes,
            "disk_write_bytes": disk_io.write_bytes,
            "network_bytes_sent": net_io.bytes_sent,
            "network_bytes_recv": net_io.bytes_recv
        }
    )

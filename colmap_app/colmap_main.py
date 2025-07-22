"""
This module defines the FastAPI application for the COLMAP CLI-based Structure from Motion (SfM) pipeline.
It includes endpoints for initiating COLMAP processing, checking job status, and monitoring compute usage.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
import os
import uuid
import asyncio
import psutil # For compute_status endpoint
import logging # Import logging module
from datetime import datetime
import json # For saving/loading JSON outputs
import cv2 # For frame extraction
from PIL import Image # For frame extraction
import numpy as np # For frame extraction
import shutil # For directory operations like rmtree

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the ColmapProcessor
from colmap_processor import ColmapProcessor

# Initialize FastAPI app
app = FastAPI(
    title="COLMAP CLI SfM API",
    description="API for processing videos to estimate 3D reconstruction using COLMAP CLI.",
    version="1.0.0"
)

# Configuration directories (relative to /app in Docker)
VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/outputs"

# In-memory storage for job statuses
job_statuses = {}

# Global ColmapProcessor instance
colmap_processor_instance: ColmapProcessor = None

# Ensure necessary data directories exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs when the FastAPI application starts up.
    Used to initialize the global ColmapProcessor instance.
    """
    global colmap_processor_instance
    logger.info("APP STARTUP: Initializing ColmapProcessor (COLMAP CLI)...")
    try:
        colmap_processor_instance = ColmapProcessor()
        logger.info("APP STARTUP: COLMAP CLI Processor loaded successfully. API is now ready.")
    except Exception as e:
        logger.error(f"APP STARTUP ERROR: Failed to load COLMAP CLI Processor: {e}", exc_info=True)
        # It's critical for this app, so raising an exception might be appropriate
        # if COLMAP CLI is truly not found. For now, we'll log and let it start.


@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    logger.info("API Request: / received.")
    return {"message": "Welcome to the COLMAP CLI SfM API."}

def update_job_progress_callback(job_id: str, progress_percent: int, stage: str):
    """
    Callback function to update the job status with real-time progress.
    """
    job_statuses[job_id].update({
        "progress": f"{progress_percent}%",
        "stage": stage,
        "message": f"Processing: {stage} ({progress_percent}%)"
    })
    logger.info(f"Job {job_id} progress: {progress_percent}% - {stage}")


async def process_colmap_background(job_id: str, input_video_path: str, output_base_path: str, sample_frames: int, use_gpu_colmap: bool, max_image_size_colmap: int):
    """
    Background task to perform Structure from Motion processing using COLMAP CLI.
    """
    global colmap_processor_instance
    if colmap_processor_instance is None:
        job_statuses[job_id].update({
            "status": "FAILED",
            "message": "COLMAP Processor not initialized. COLMAP CLI might not be installed or accessible.",
            "error": "COLMAP CLI not ready.",
            "progress": "Error",
            "stage": "Initialization Failed"
        })
        logger.error(f"Job {job_id}: COLMAP Processor not ready, cannot process.")
        return

    job_statuses[job_id] = {"status": "PROCESSING", "message": "COLMAP job initialized.", "progress": "0%", "stage": "Starting"}
    logger.info(f"Job {job_id}: Starting COLMAP CLI processing for {input_video_path}")

    temp_frames_dir = os.path.join("/tmp", f"colmap_frames_{job_id}")
    
    try:
        # Stage 1: Frame Extraction
        update_job_progress_callback(job_id, 10, "Extracting Frames for COLMAP")
        logger.info(f"Job {job_id}: Starting frame extraction for COLMAP with sampling {sample_frames} frames.")
        
        # Use asyncio.to_thread for blocking OpenCV operations
        def _blocking_sampled_frame_extraction():
            frames_output_path = os.path.join(temp_frames_dir, "images")
            os.makedirs(frames_output_path, exist_ok=True)

            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_video_path}")

            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_video_frames == 0:
                raise ValueError("Video contains no frames.")

            num_frames_to_sample = min(sample_frames, total_video_frames)
            
            if num_frames_to_sample > 1:
                frame_indices = np.linspace(0, total_video_frames - 1, num_frames_to_sample, dtype=int)
            else:
                frame_indices = [0] if total_video_frames > 0 else []

            saved_frame_count = 0
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Job {job_id}: Could not read frame at index {frame_idx}. Skipping.")
                    continue
                
                frame_filename = os.path.join(frames_output_path, f"{saved_frame_count:06d}.png")
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_filename)
                saved_frame_count += 1
            cap.release()
            return frames_output_path, saved_frame_count

        image_folder, saved_frame_count = await asyncio.to_thread(_blocking_sampled_frame_extraction)
        if saved_frame_count == 0:
            raise RuntimeError("No frames were extracted from the video for COLMAP.")
        
        logger.info(f"Job {job_id}: Extracted {saved_frame_count} sampled frames for COLMAP to {image_folder}.")
        update_job_progress_callback(job_id, 30, "Frames Extracted for COLMAP")


        # Stage 2: Run COLMAP CLI pipeline
        update_job_progress_callback(job_id, 40, "Running COLMAP Reconstruction")
        colmap_output_files = await colmap_processor_instance.process_images(
            image_folder=image_folder,
            output_base_path=output_base_path,
            job_id=job_id,
            use_gpu=use_gpu_colmap,
            max_image_size=max_image_size_colmap
            # krt_coords_path removed
        )
        update_job_progress_callback(job_id, 90, "COLMAP Reconstruction Complete")

        job_statuses[job_id].update({
            "status": "COMPLETED",
            "message": "COLMAP SfM processing completed successfully.",
            "output_files": colmap_output_files,
            "progress": "100%",
            "stage": "Completed"
        })
        logger.info(f"Job {job_id}: COLMAP SfM processing completed. Outputs: {colmap_output_files}")

    except Exception as e:
        job_statuses[job_id].update({
            "status": "FAILED",
            "message": "An error occurred during COLMAP SfM processing.",
            "error": str(e),
            "progress": "Error",
            "stage": "Failed"
        })
        logger.error(f"Job {job_id}: COLMAP SfM processing failed: {e}", exc_info=True)
    finally:
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
            logger.info(f"Job {job_id}: Cleaned up input video file {input_video_path}")
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)
            logger.info(f"Job {job_id}: Cleaned up temporary COLMAP frames directory: {temp_frames_dir}")


@app.post("/colmap_predict")
async def colmap_predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_name: str = Form(None, description="Optional: Base name for the output files. Defaults to a timestamped name."),
    sample_frames: int = Form(50, description="Number of frames to sample from the video. Defaults to 50."),
    use_gpu: bool = Form(True, description="Whether to use GPU for COLMAP processing. Highly recommended for performance."),
    max_image_size: int = Form(4000, description="Maximum image size for COLMAP undistortion to prevent downsampling. Set to a high value like 4000 or more to avoid downsampling.")
    # krt_coords_path removed
):
    """
    Initiates an asynchronous job to perform Structure from Motion on an uploaded video using COLMAP CLI.
    """
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    final_output_name = output_name if output_name else f"colmap_sfm_run_{timestamp}"
    
    input_video_path = os.path.join(VIDEO_DIR, f"{final_output_name}_{file.filename}")
    output_base_path = os.path.join(OUTPUT_DIR, final_output_name)

    logger.info(f"API Request: /colmap_predict (COLMAP CLI) received for job {job_id}")

    if colmap_processor_instance is None:
        logger.warning(f"API Request: /colmap_predict (COLMAP CLI) rejected - COLMAP Processor not initialized.")
        raise HTTPException(status_code=503, detail="Service Unavailable: COLMAP CLI is not installed or accessible at startup.")

    try:
        logger.info(f"Job {job_id}: Saving uploaded video to {input_video_path}")
        with open(input_video_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"Job {job_id}: Video saved.")

        job_statuses[job_id] = {"status": "PENDING", "message": "COLMAP SfM job received and queued.", "progress": "0%", "stage": "Queued"}

        background_tasks.add_task(
            process_colmap_background,
            job_id,
            input_video_path,
            output_base_path,
            sample_frames,
            use_gpu,
            max_image_size
            # krt_coords_path removed
        )
        logger.info(f"Job {job_id}: COLMAP Background task added.")

        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": "COLMAP SfM processing job started. Check status using the job ID.",
                "job_id": job_id,
                "output_name": final_output_name
            }
        )
    except Exception as e:
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
        if job_id in job_statuses:
            del job_statuses[job_id]
        logger.error(f"API Request: /colmap_predict (COLMAP CLI) failed for job {job_id}. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initiate COLMAP SfM processing: {str(e)}")


@app.get("/sfm_status/{job_id}")
async def sfm_status(job_id: str):
    """
    Checks the current status of an SfM processing job.
    """
    logger.info(f"API Request: /sfm_status/{job_id} received.")
    status_info = job_statuses.get(job_id)
    if status_info:
        response_content = status_info
        headers = {}
        if status_info["status"] in ["PENDING", "PROCESSING"]:
            headers["Retry-After"] = "5"
        logger.info(f"API Request: /sfm_status/{job_id} returning status: {status_info['status']}")
        return JSONResponse(status_code=200, content=response_content, headers=headers)
    else:
        logger.warning(f"API Request: /sfm_status/{job_id} - Job ID not found.")
        raise HTTPException(status_code=404, detail="Job ID not found.")


@app.get("/compute_status")
async def compute_status():
    """
    Provides real-time information about the Docker container's CPU, memory, disk I/O, and network I/O usage.
    """
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()

    logger.info("API Request: /compute_status received. Returning system metrics.")

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

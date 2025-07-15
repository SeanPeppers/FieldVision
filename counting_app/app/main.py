"""
This module provides a FastAPI application for object counting in videos.
It exposes a /predict endpoint to process video files asynchronously
and a /status endpoint to check the job progress, including real-time updates.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
from datetime import datetime
import uuid # For generating unique job IDs
from app.models import ObjectCounterModel
import ultralytics
from ultralytics.utils.downloads import safe_download
import cv2
import math # For ceiling division if needed for progress calculation
import asyncio # Import asyncio to use to_thread for blocking operations

# Initialize FastAPI app
app = FastAPI(
    title="YOLO11 Object Counting API",
    description="API for counting objects in videos using Ultralytics YOLO11 and asynchronous processing with real-time progress.",
    version="1.0.0"
)

# Configuration for the model
MODEL_PATH = "yolo11n.pt"  # Pre-trained YOLO11 Nano model
OUTPUT_DIR = "data/output"
VIDEO_DIR = "data/videos"

# Configuration for the counting region
# This factor determines the vertical position (from the top) where the counting region starts.
# E.g., 0.5 for the middle, 0.6 for 60% down, 0.75 for 75% down.
REGION_VERTICAL_OFFSET_FACTOR = 0.6 # Adjust this value (0.0 to 1.0) to move the region up/down

# In-memory storage for job statuses
# NOTE: This will not persist across API restarts. For production, use a database (e.g., Redis, PostgreSQL).
job_statuses = {}

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model: {MODEL_PATH}...")
    try:
        # This will attempt to download if yolo11n.pt is not found locally
        _ = ultralytics.YOLO(MODEL_PATH)
        print(f"Model {MODEL_PATH} downloaded successfully.")
    except Exception as e:
        print(f"Could not automatically download {MODEL_PATH}. Please ensure it's available or manually download it. Error: {e}")
        # As a fallback, try to download a sample video if model download fails for demonstration
        safe_download("https://github.com/ultralytics/notebooks/releases/download/v0.0.0/solutions-ci-demo.mp4", VIDEO_DIR)
        print("Downloaded sample video 'solutions-ci-demo.mp4' to 'data/videos'.")


@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the YOLO11 Object Counting API. Use /predict to process videos."}

def update_progress_callback(job_id: str, current_frame: int, total_frames: int):
    """
    Callback function to update the job status with real-time progress.
    This function is called by ObjectCounterModel.process_video.
    """
    if total_frames > 0:
        progress_percent = min(100, int((current_frame / total_frames) * 100))
        job_statuses[job_id]["progress"] = f"{progress_percent}%"
        job_statuses[job_id]["current_frame"] = current_frame
        job_statuses[job_id]["total_frames"] = total_frames
        # print(f"Job {job_id} progress: {progress_percent}% ({current_frame}/{total_frames})") # Optional: for server logs
    else:
        job_statuses[job_id]["progress"] = "0%"


async def process_video_background(job_id: str, input_video_path: str, output_video_path: str):
    """
    Background task to process the video for object counting.
    Updates the job status in the global job_statuses dictionary.
    This function now runs the blocking video processing in a separate thread.
    """
    job_statuses[job_id] = {"status": "PROCESSING", "message": "Video processing in progress.", "progress": "0%"}
    print(f"Job {job_id}: Starting video processing for {input_video_path}")

    try:
        # Get video dimensions to calculate dynamic region points
        # This part is still synchronous, but generally fast.
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file to determine dimensions.")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release() # Release the capture immediately after getting dimensions

        # Define region points as a rectangle covering the area from REGION_VERTICAL_OFFSET_FACTOR down to the bottom
        line_y = int(h * REGION_VERTICAL_OFFSET_FACTOR)
        region_points = [(0, line_y), (w, line_y), (w, h), (0, h)]

        # Initialize the object counter model for this specific video
        object_counter_model = ObjectCounterModel(
            model_path=MODEL_PATH,
            region_points=region_points, # Use dynamically calculated region points (now a rectangle)
            show_display=False,
        )

        # Process the video with the object counter model, passing the progress callback
        # Use asyncio.to_thread to run the blocking process_video method in a separate thread
        success = await asyncio.to_thread(
            object_counter_model.process_video,
            video_path=input_video_path,
            output_path=output_video_path,
            progress_callback=lambda current, total: update_progress_callback(job_id, current, total)
        )

        if success:
            job_statuses[job_id].update({
                "status": "COMPLETED",
                "message": "Object counting completed successfully.",
                "output_video_location": output_video_path,
                "progress": "100%" # Ensure progress is 100% on completion
            })
            print(f"Job {job_id}: Video processing completed. Output: {output_video_path}")
        else:
            job_statuses[job_id].update({
                "status": "FAILED",
                "message": "Failed to process video for object counting.",
                "error": "Unknown error during video processing."
            })
            print(f"Job {job_id}: Video processing failed.")

    except Exception as e:
        job_statuses[job_id].update({
            "status": "FAILED",
            "message": "An error occurred during video processing.",
            "error": str(e),
            "progress": "Error" # Indicate error in progress
        })
        print(f"Job {job_id}: An error occurred: {e}")
    finally:
        # Clean up the input video file after processing (success or failure)
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
            print(f"Job {job_id}: Cleaned up input file {input_video_path}")


@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Predict endpoint to initiate object counting on an uploaded video file as a background task.

    Args:
        background_tasks (BackgroundTasks): FastAPI dependency for managing background tasks.
        file (UploadFile): The video file to process.

    Returns:
        JSONResponse: A JSON response indicating that the job has been accepted, along with a job ID.
    """
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_video_path = os.path.join(VIDEO_DIR, f"{timestamp}_{file.filename}")
    output_video_path = os.path.join(OUTPUT_DIR, f"counted_{timestamp}_{file.filename}")

    try:
        # Save the uploaded video file
        with open(input_video_path, "wb") as buffer:
            buffer.write(await file.read())

        # Initialize job status
        job_statuses[job_id] = {"status": "PENDING", "message": "Job received and queued.", "progress": "0%"}

        # Add the video processing to background tasks
        background_tasks.add_task(process_video_background, job_id, input_video_path, output_video_path)

        return JSONResponse(
            status_code=202, # 202 Accepted
            content={
                "status": "accepted",
                "message": "Video processing job started. Check status using the job ID.",
                "job_id": job_id
            }
        )
    except Exception as e:
        # Clean up the uploaded file if an error occurs during initial file save
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
        # Remove job from status if it failed before even starting background task
        if job_id in job_statuses:
            del job_statuses[job_id]
        raise HTTPException(status_code=500, detail=f"Failed to initiate video processing: {str(e)}")


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Endpoint to check the status of a video processing job.

    Args:
        job_id (str): The unique ID of the job to check.

    Returns:
        JSONResponse: A JSON response containing the current status of the job,
                      and a 'Retry-After' header if the job is still processing.
    """
    status_info = job_statuses.get(job_id)
    if status_info:
        response_content = status_info
        headers = {}
        if status_info["status"] in ["PENDING", "PROCESSING"]:
            # Suggest polling every 5 seconds while processing
            headers["Retry-After"] = "5"
        return JSONResponse(status_code=200, content=response_content, headers=headers)
    else:
        raise HTTPException(status_code=404, detail="Job ID not found.")


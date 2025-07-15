"""
This module provides a FastAPI application for object counting in videos.
It exposes a /predict endpoint to process video files asynchronously
and a /status endpoint to check the job progress, including real-time updates.
A new /finetune endpoint is added to allow asynchronous fine-tuning of the model.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
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
import yaml # For parsing data.yaml

# Initialize FastAPI app
app = FastAPI(
    title="YOLO11 Object Counting API",
    description="API for counting objects in videos using Ultralytics YOLO11 and asynchronous processing with real-time progress. Includes model fine-tuning capability.",
    version="1.0.0"
)

# Configuration for the base model
# This will always be the default model path for predictions unless overridden.
DEFAULT_MODEL_PATH = "yolo11n.pt"
OUTPUT_DIR = "data/output"
VIDEO_DIR = "data/videos"
DATASET_DIR = "data/datasets" # Directory for downloaded datasets
FINE_TUNED_MODELS_DIR = "data/finetuned_models" # Directory for fine-tuned models

# Configuration for the counting region
# This factor determines the vertical position (from the top) where the counting region starts.
# E.g., 0.5 for the middle, 0.6 for 60% down, 0.75 for 75% down.
REGION_VERTICAL_OFFSET_FACTOR = 0.6 # Adjust this value (0.0 to 1.0) to move the region up/down

# In-memory storage for job statuses
# NOTE: This will not persist across API restarts. For production, use a database (e.g., Redis, PostgreSQL).
job_statuses = {}

# The specific class name you want to count during inference (e.g., "maize")
# This should match a class name in your fine-tuned model's data.yaml
TARGET_INFERENCE_CLASS_NAME = "maize" # <--- CONFIGURE THIS TO YOUR TARGET CLASS NAME

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(FINE_TUNED_MODELS_DIR, exist_ok=True)

# Download the base model if it doesn't exist
if not os.path.exists(DEFAULT_MODEL_PATH):
    print(f"Downloading base model: {DEFAULT_MODEL_PATH}...")
    try:
        # This will attempt to download if yolo11n.pt is not found locally
        _ = ultralytics.YOLO(DEFAULT_MODEL_PATH)
        print(f"Model {DEFAULT_MODEL_PATH} downloaded successfully.")
    except Exception as e:
        print(f"Could not automatically download {DEFAULT_MODEL_PATH}. Please ensure it's available or manually download it. Error: {e}")
        # As a fallback, try to download a sample video if model download fails for demonstration
        safe_download("https://github.com/ultralytics/notebooks/releases/download/v0.0.0/solutions-ci-demo.mp4", VIDEO_DIR)
        print("Downloaded sample video 'solutions-ci-demo.mp4' to 'data/videos'.")


@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the YOLO11 Object Counting API. Use /predict to process videos or /finetune to fine-tune the model."}

def update_progress_callback(job_id: str, current_frame: int, total_frames: int):
    """
    Callback function to update the job status with real-time progress for video processing.
    This function is called by ObjectCounterModel.process_video.
    """
    if total_frames > 0:
        progress_percent = min(100, int((current_frame / total_frames) * 100))
        job_statuses[job_id]["progress"] = f"{progress_percent}%"
        job_statuses[job_id]["current_frame"] = current_frame
        job_statuses[job_id]["total_frames"] = total_frames
    else:
        job_statuses[job_id]["progress"] = "0%"

def update_finetune_progress_callback(job_id: str, epoch: int, total_epochs: int, metrics: dict = None):
    """
    Callback function to update the job status with real-time progress for fine-tuning.
    """
    if total_epochs > 0:
        progress_percent = min(100, int((epoch / total_epochs) * 100))
        job_statuses[job_id]["progress"] = f"{progress_percent}%"
        job_statuses[job_id]["current_epoch"] = epoch
        job_statuses[job_id]["total_epochs"] = total_epochs
        if metrics:
            job_statuses[job_id]["metrics"] = metrics
    else:
        job_statuses[job_id]["progress"] = "0%"


async def process_video_background(job_id: str, input_video_path: str, output_video_path: str, model_to_use_path: str):
    """
    Background task to process the video for object counting.
    Updates the job status in the global job_statuses dictionary.
    This function now runs the blocking video processing in a separate thread.
    """
    job_statuses[job_id] = {"status": "PROCESSING", "message": "Video processing in progress.", "progress": "0%"}
    print(f"Job {job_id}: Starting video processing for {input_video_path} using model: {model_to_use_path}")

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

        # Determine classes to count based on the selected model's data.yaml
        classes_to_count = None
        current_model_class_names = {}

        # If a fine-tuned model is being used, try to load its specific data.yaml
        if model_to_use_path != DEFAULT_MODEL_PATH:
            # Assume data.yaml is alongside the weights or in the parent directory of weights
            # Ultralytics typically copies data.yaml to the run directory (e.g., data/finetuned_models/model_name/data.yaml)
            model_run_dir = os.path.dirname(os.path.dirname(model_to_use_path)) # Go up two levels from best.pt
            model_data_yaml_path = os.path.join(model_run_dir, 'data.yaml')

            if os.path.exists(model_data_yaml_path):
                try:
                    with open(model_data_yaml_path, 'r') as f:
                        model_data_yaml_content = yaml.safe_load(f)
                    
                    loaded_class_names_list = model_data_yaml_content.get('names')
                    if loaded_class_names_list:
                        current_model_class_names = {name: idx for idx, name in enumerate(loaded_class_names_list)}
                        print(f"Job {job_id}: Loaded class names from {model_data_yaml_path}: {current_model_class_names}")
                    else:
                        print(f"Job {job_id}: Warning: 'names' not found in {model_data_yaml_path}.")
                except Exception as e:
                    print(f"Job {job_id}: Warning: Could not parse {model_data_yaml_path}. Error: {e}")
            else:
                print(f"Job {job_id}: Warning: data.yaml not found for fine-tuned model at {model_data_yaml_path}. Attempting to load classes from model object directly.")
                # Fallback to loading from model object if data.yaml not found
                try:
                    temp_model = ultralytics.YOLO(model_to_use_path)
                    if hasattr(temp_model, 'names') and temp_model.names:
                        current_model_class_names = {name: idx for idx, name in temp_model.names.items()}
                    print(f"Job {job_id}: Loaded class names directly from model {model_to_use_path}: {current_model_class_names}")
                except Exception as e:
                    print(f"Job {job_id}: Warning: Could not load class names directly from model {model_to_use_path}. Error: {e}")

        else: # Using DEFAULT_MODEL_PATH
            print(f"Job {job_id}: Using default model. Attempting to load COCO class names.")
            try:
                temp_model = ultralytics.YOLO(model_to_use_path)
                if hasattr(temp_model, 'names') and temp_model.names:
                    current_model_class_names = {name: idx for idx, name in temp_model.names.items()}
                print(f"Job {job_id}: Loaded class names for default model: {current_model_class_names}")
            except Exception as e:
                print(f"Job {job_id}: Warning: Could not load class names from default model {model_to_use_path}. Error: {e}")


        # Apply TARGET_INFERENCE_CLASS_NAME filter
        if TARGET_INFERENCE_CLASS_NAME and current_model_class_names:
            target_class_id = current_model_class_names.get(TARGET_INFERENCE_CLASS_NAME)
            if target_class_id is not None:
                classes_to_count = [target_class_id]
                print(f"Job {job_id}: Counting only class '{TARGET_INFERENCE_CLASS_NAME}' (ID: {target_class_id}) for prediction.")
            else:
                print(f"Job {job_id}: Warning: Target inference class '{TARGET_INFERENCE_CLASS_NAME}' not found in selected model's classes. Counting all classes from the selected model.")
                # If target class not found, count ALL classes that the selected model knows about
                classes_to_count = list(current_model_class_names.values())
        elif current_model_class_names:
            # If no specific TARGET_INFERENCE_CLASS_NAME is set, but we have model classes, count all of them.
            print(f"Job {job_id}: No specific target class configured. Counting all classes from the selected model.")
            classes_to_count = list(current_model_class_names.values())
        else:
            # Fallback if no class names could be loaded at all
            print(f"Job {job_id}: Critical Warning: No model class names loaded. Counting all detected classes without specific filtering.")


        # Initialize the object counter model for this specific video
        object_counter_model = ObjectCounterModel(
            model_path=model_to_use_path, # Use the dynamically determined model path
            region_points=region_points, # Use dynamically calculated region points (now a rectangle)
            show_display=False,
            classes=classes_to_count # Pass the specific class ID(s) if found, otherwise None (all)
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


async def finetune_model_background(
    job_id: str,
    dataset_download_url: str,
    epochs: int,
    model_name: str
):
    """
    Background task to fine-tune the YOLO model.
    Updates the job status in the global job_statuses dictionary.
    """
    # MODEL_PATH is no longer updated globally here.
    # It remains DEFAULT_MODEL_PATH unless explicitly chosen in /predict.

    job_statuses[job_id] = {"status": "FINETUNING_PENDING", "message": "Fine-tuning job queued.", "progress": "0%"}
    print(f"Job {job_id}: Starting fine-tuning with dataset from {dataset_download_url}")

    dataset_zip_path = None
    extracted_dataset_path = None
    temp_extracted_dir = None # Initialize to None for cleanup in finally block
    try:
        # 1. Download Dataset using curl
        job_statuses[job_id].update({"status": "DOWNLOADING_DATASET", "message": "Downloading dataset...", "progress": "0%"})
        print(f"Job {job_id}: Downloading dataset from {dataset_download_url}")

        # Define paths for download and extraction
        dataset_zip_filename = f"roboflow_dataset_{job_id}.zip"
        dataset_zip_path = os.path.join(DATASET_DIR, dataset_zip_filename)
        # The extraction will go into a temporary directory, and we'll find the actual dataset root inside it.
        temp_extracted_dir = os.path.join(DATASET_DIR, f"temp_extracted_{job_id}")
        os.makedirs(temp_extracted_dir, exist_ok=True) # Ensure temporary extraction directory exists

        # Construct the curl command
        curl_command = f'curl -L "{dataset_download_url}" > "{dataset_zip_path}"'
        print(f"Job {job_id}: Executing download command: {curl_command}")
        download_exit_code = await asyncio.to_thread(os.system, curl_command)

        if download_exit_code != 0:
            raise RuntimeError(f"Dataset download failed with exit code {download_exit_code}")
        if not os.path.exists(dataset_zip_path):
            raise FileNotFoundError(f"Dataset download failed: zip file not found at {dataset_zip_path}")

        # 2. Unzip Dataset
        job_statuses[job_id].update({"status": "EXTRACTING_DATASET", "message": "Extracting dataset...", "progress": "0%"})
        print(f"Job {job_id}: Extracting dataset from {dataset_zip_path} to {temp_extracted_dir}")

        # Unzip the file into the temporary extracted directory
        unzip_command = f'unzip "{dataset_zip_path}" -d "{temp_extracted_dir}"'
        print(f"Job {job_id}: Executing unzip command: {unzip_command}")
        unzip_exit_code = await asyncio.to_thread(os.system, unzip_command)

        if unzip_exit_code != 0:
            raise RuntimeError(f"Dataset extraction failed with exit code {unzip_exit_code}")

        # 3. Find the data.yaml file directly within the temporary extracted directory
        # Based on the screenshot, data.yaml is at the root of the extracted content.
        data_yaml_path = os.path.join(temp_extracted_dir, 'data.yaml')
        
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}. Ensure correct dataset structure.")
        
        print(f"Job {job_id}: Dataset extracted to: {temp_extracted_dir}")
        print(f"Job {job_id}: Found data.yaml at: {data_yaml_path}")

        # Load data.yaml to get class names (for job status reporting)
        with open(data_yaml_path, 'r') as f:
            data_yaml_content = yaml.safe_load(f)
        
        dataset_class_names = data_yaml_content.get('names')
        if not dataset_class_names:
            raise ValueError("Could not find 'names' (class names) in data.yaml.")
        
        # Store class names in job status for reference, but not globally for inference
        job_statuses[job_id]["class_names_trained"] = {name: idx for idx, name in enumerate(dataset_class_names)}
        print(f"Job {job_id}: Loaded class names from data.yaml: {job_statuses[job_id]['class_names_trained']}")

        # 4. Fine-tune Model
        job_statuses[job_id].update({"status": "FINETUNING_MODEL", "message": "Starting model training...", "progress": "0%"})
        print(f"Job {job_id}: Starting model fine-tuning for {epochs} epochs.")

        # Load the base model for fine-tuning (always start from DEFAULT_MODEL_PATH)
        model = ultralytics.YOLO(DEFAULT_MODEL_PATH)

        # Define a simple callback for training progress
        def training_callback(trainer):
            # This callback runs after each epoch
            if hasattr(trainer, 'epoch') and hasattr(trainer, 'epochs'):
                current_epoch = trainer.epoch + 1 # epoch is 0-indexed
                total_epochs = trainer.epochs
                metrics = {}
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    # Access relevant metrics if available, e.g., mAP, loss
                    metrics = {k: v for k, v in trainer.metrics.items() if isinstance(v, (int, float))}
                update_finetune_progress_callback(job_id, current_epoch, total_epochs, metrics)
                # print(f"Job {job_id} Fine-tuning progress: Epoch {current_epoch}/{total_epochs}, Metrics: {metrics}") # For server logs

        # Add the callback to the model's training process
        model.add_callback("on_train_epoch_end", training_callback)

        # Perform the training in a separate thread as it's blocking
        results = await asyncio.to_thread(
            model.train,
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640, # Common image size for YOLO, adjust if your dataset requires
            project=FINE_TUNED_MODELS_DIR,
            name=model_name,
            val=False, # Explicitly set to False as requested
            # Add other training parameters as needed (e.g., batch, lr, etc.)
        )

        # Construct the path to the fine-tuned model
        fine_tuned_model_run_dir = os.path.join(FINE_TUNED_MODELS_DIR, model_name)
        final_model_path = os.path.join(fine_tuned_model_run_dir, 'weights', 'best.pt')

        if not os.path.exists(final_model_path):
            final_model_path = os.path.join(fine_tuned_model_run_dir, 'weights', 'last.pt')
            if not os.path.exists(final_model_path):
                raise FileNotFoundError(f"Fine-tuned model (best.pt or last.pt) not found at {fine_tuned_model_run_dir}")

        # The global MODEL_PATH is NOT updated here.
        # It remains DEFAULT_MODEL_PATH.
        # The path to the fine-tuned model is returned in the job status.
        print(f"Job {job_id}: Fine-tuned model saved to: {final_model_path}")

        job_statuses[job_id].update({
            "status": "COMPLETED",
            "message": "Model fine-tuning completed successfully. Use the 'model_name' parameter in /predict to use this model.",
            "fine_tuned_model_path": final_model_path,
            "progress": "100%"
        })
        print(f"Job {job_id}: Model fine-tuning completed.")

    except Exception as e:
        job_statuses[job_id].update({
            "status": "FAILED",
            "message": "An error occurred during model fine-tuning.",
            "error": str(e),
            "progress": "Error"
        })
        print(f"Job {job_id}: Fine-tuning failed: {e}")
    finally:
        # Clean up downloaded zip file (commented out for debugging)
        # if dataset_zip_path and os.path.exists(dataset_zip_path):
        #     await asyncio.to_thread(os.remove, dataset_zip_path)
        #     print(f"Job {job_id}: Cleaned up dataset zip file: {dataset_zip_path}")
        # Clean up extracted temporary dataset directory (commented out for debugging)
        # if temp_extracted_dir and os.path.exists(temp_extracted_dir):
        #     await asyncio.to_thread(os.system, f"rm -rf {temp_extracted_dir}")
        #     print(f"Job {job_id}: Cleaned up temporary extracted dataset directory: {temp_extracted_dir}")
        pass # Keep pass to explicitly indicate no cleanup for debugging


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str = Form(None, description="Optional: Name of a fine-tuned model to use for prediction (e.g., 'maize_plant_detector'). If not provided, the default YOLO11n model will be used.")
):
    """
    Predict endpoint to initiate object counting on an uploaded video file as a background task.

    Args:
        background_tasks (BackgroundTasks): FastAPI dependency for managing background tasks.
        file (UploadFile): The video file to process.
        model_name (str, optional): The name of a fine-tuned model to use. If omitted, the default
                                    'yolo11n.pt' model will be used.

    Returns:
        JSONResponse: A JSON response indicating that the job has been accepted, along with a job ID.
    """
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_video_path = os.path.join(VIDEO_DIR, f"{timestamp}_{file.filename}")
    output_video_path = os.path.join(OUTPUT_DIR, f"counted_{timestamp}_{file.filename}")

    # Determine which model path to use for this prediction
    model_to_use_path = DEFAULT_MODEL_PATH
    if model_name:
        candidate_model_path = os.path.join(FINE_TUNED_MODELS_DIR, model_name, 'weights', 'best.pt')
        if os.path.exists(candidate_model_path):
            model_to_use_path = candidate_model_path
            print(f"Job {job_id}: Using fine-tuned model: {model_to_use_path}")
        else:
            print(f"Job {job_id}: Warning: Fine-tuned model '{model_name}' not found at {candidate_model_path}. Falling back to default model: {DEFAULT_MODEL_PATH}")
            # Optionally, you could raise an HTTPException here if you want to strictly enforce
            # that a requested fine-tuned model must exist.
    else:
        print(f"Job {job_id}: Using default model: {DEFAULT_MODEL_PATH}")


    try:
        # Save the uploaded video file
        with open(input_video_path, "wb") as buffer:
            buffer.write(await file.read())

        # Initialize job status
        job_statuses[job_id] = {"status": "PENDING", "message": "Job received and queued.", "progress": "0%"}

        # Add the video processing to background tasks, passing the selected model path
        background_tasks.add_task(process_video_background, job_id, input_video_path, output_video_path, model_to_use_path)

        return JSONResponse(
            status_code=202, # 202 Accepted
            content={
                "status": "accepted",
                "message": "Video processing job started. Check status using the job ID.",
                "job_id": job_id,
                "model_name_used": model_name if model_name else "default" # Include the model name used
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


@app.post("/finetune")
async def finetune(
    background_tasks: BackgroundTasks,
    dataset_download_url: str = Form(..., description="Public URL of the dataset (e.g., Roboflow direct download link)."),
    epochs: int = Form(5, description="Number of training epochs for fine-tuning."),
    model_name: str = Form("custom_yolo_model", description="Name for the fine-tuned model (will create a directory under data/finetuned_models).")
):
    """
    Endpoint to initiate model fine-tuning as a background task.
    Downloads the dataset using curl and fine-tunes the YOLO model.

    Args:
        background_tasks (BackgroundTasks): FastAPI dependency for managing background tasks.
        dataset_download_url (str): Public URL of the dataset (e.g., Roboflow direct download link).
                                    The dataset should be a .zip file in YOLO-compatible format (images and labels, with a data.yaml).
        epochs (int): Number of training epochs for fine-tuning.
        model_name (str): A unique name for the fine-tuned model. This will be used to create a directory
                          under `data/finetuned_models` to store the training results.

    Returns:
        JSONResponse: A JSON response indicating that the fine-tuning job has been accepted, along with a job ID and the model name.
    """
    job_id = str(uuid.uuid4())
    job_statuses[job_id] = {"status": "PENDING", "message": "Fine-tuning job received and queued.", "progress": "0%"}

    background_tasks.add_task(
        finetune_model_background,
        job_id,
        dataset_download_url,
        epochs,
        model_name
    )

    return JSONResponse(
        status_code=202, # 202 Accepted
        content={
            "status": "accepted",
            "message": "Model fine-tuning job started. Check status using the job ID.",
            "job_id": job_id,
            "model_name": model_name # Include the model name in the initial response
        }
    )


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Endpoint to check the status of a video processing or model fine-tuning job.

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
        if status_info["status"] in ["PENDING", "PROCESSING", "DOWNLOADING_DATASET", "EXTRACTING_DATASET", "FINETUNING_PENDING", "FINETUNING_PROCESSING", "FINETUNING_MODEL"]:
            # Suggest polling every 5 seconds while processing
            headers["Retry-After"] = "5"
        return JSONResponse(status_code=200, content=response_content, headers=headers)
    else:
        raise HTTPException(status_code=404, detail="Job ID not found.")

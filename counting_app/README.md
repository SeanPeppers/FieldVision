# YOLO11 Object Counting API

This project provides a production-ready FastAPI application for performing object counting in video files using Ultralytics YOLO11. It allows you to upload a video, process it with a pre-trained YOLO model, and receive an output video with detected and counted objects.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
  - [Local Setup (Without Docker)](#local-setup-without-docker)
  - [Docker Setup](#docker-setup)
- [API Usage](#api-usage)
  - [Root Endpoint](#root-endpoint)
  - [Predict Endpoint](#predict-endpoint)
  - [Fine-tune Endpoint](#fine-tune-endpoint)
  - [Status Endpoint](#status-endpoint)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

*(Add details about the project structure here.)*

---

## Features

- **FastAPI Backend**: A robust and high-performance API built with FastAPI.
- **YOLO11 Integration**: Utilizes Ultralytics YOLO11 for efficient object detection and counting.
- **Video Processing**: Processes uploaded video files frame by frame.
- **Configurable Counting Region**: Define a specific region of interest for object counting.
- **Docker Support**: Easily containerize and deploy the application using Docker.
- **Structured Output**: Provides a clear JSON response with the status and output video location.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9+
- `pip` (Python package installer)
- `git` (Optional, for cloning the repository)
- Docker (Optional, if using Docker for deployment)

---

## Setup and Installation

You can run this application either directly on your machine or using Docker.

### Local Setup (Without Docker)

1. **Clone the repository** (if applicable):

    ```bash
    git clone <repository_url>
    cd counting_app
    ```

    *(Or navigate to the `counting_app` directory if you already have the files.)*

2. **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the YOLO model**:

    The app will automatically attempt to download `yolo11n.pt` on the first run.

5. **Prepare data directories**:

    Ensure the `data/videos` and `data/output` directories exist. They will be created on startup if not found.

6. **Run the application**:

    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

    The API will be available at: [http://localhost:8000](http://localhost:8000)

---

### Docker Setup

1. **Ensure Docker is installed and running**.

2. **Navigate to the project directory**:

    ```bash
    cd counting_app
    ```

3. **Build the Docker image**:

    ```bash
    docker build -t yolo-counting-api:latest .
    ```

4. **Run the Docker container**:

    **Linux/macOS**:

    ```bash
    docker run -d -p 8000:8000 \
        -v "$(pwd)/data/videos:/app/data/videos" \
        -v "$(pwd)/data/output:/app/data/output" \
        --name yolo-counter yolo-counting-api:latest
    ```

    **Windows (PowerShell)**:

    ```powershell
    docker run -d -p 8000:8000 `
        -v "${PWD}/data/videos:/app/data/videos" `
        -v "${PWD}/data/output:/app/data/output" `
        --name yolo-counter yolo-counting-api:latest
    ```

5. **Verify the container is running**:

    ```bash
    docker ps
    ```

    You should see `yolo-counter` listed.

---

## API Usage

Once the Docker container is running, the API is accessible at [http://localhost:8000](http://localhost:8000). View the interactive Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).

### Root Endpoint

Confirms the API is running.

- **URL**: `/`
- **Method**: `GET`
- **Response**:

    ```json
    {
      "message": "Welcome to the YOLO11 Object Counting API. Use /predict to process videos or /finetune to fine-tune the model."
    }
    ```

---

### Predict Endpoint

Initiates asynchronous object counting from an uploaded video, optionally using a fine-tuned model.

- **URL**: `/predict`
- **Method**: `POST`
- **Parameters**:
  - `file` (File, required): Video file to upload.
  - `model_name` (Form, optional): Name of a fine-tuned model (e.g., `maize_plant_detector`). Defaults to `yolo11n.pt` if not provided.
- **Response** (202 Accepted):

    ```json
    {
      "status": "accepted",
      "message": "Video processing job started. Check status using the job ID.",
      "job_id": "a1b2c3d4-e5f6-7899-1234-567890abcdef",
      "model_name_used": "maize_plant_detector" // Or "default" if no model_name was provided
    }
    ```

#### Example Usage (using `curl`)

Using the default model:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/video.mp4"
```

(Replace `/path/to/your/video.mp4` with the actual path to your video file on your host machine.)

Using a fine-tuned model (e.g., `maize_plant_detector`):

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/video.mp4" \
     -F "model_name=maize_plant_detector"
```

---

### Fine-tune Endpoint

Initiates asynchronous YOLO model fine-tuning with a custom dataset.

- **URL**: `/finetune`
- **Method**: `POST`
- **Parameters** (Form):
  - `dataset_download_url` (string, required): Public URL of YOLO-compatible dataset (.zip with data.yaml at root).
  - `epochs` (integer, optional, default: 5): Number of training epochs.
  - `model_name` (string, optional, default: 'custom_yolo_model'): Unique name for the fine-tuned model, stored under data/finetuned_models/.
- **Response** (202 Accepted):

    ```json
    {
      "status": "accepted",
      "message": "Model fine-tuning job started. Check status using the job ID.",
      "job_id": "e5f6-7890-1234-567890abcdef-a1b2c3d4",
      "model_name": "maize_plant_detector" # The name provided for the fine-tuned model
    }
    ```

#### Example Usage (using `curl`)

```bash
curl -X POST "http://localhost:8000/finetune" \
     -H "accept: application/json" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "dataset_download_url=https://app.roboflow.com/ds/KEC0zfNa50?key=vD5a0QNRsA" \
     -d "epochs=5" \
     -d "model_name=maize_plant_detector"
```

(Replace the `dataset_download_url` with your actual Roboflow direct download link.)

---

### Status Endpoint

Checks the current status of a video processing or model fine-tuning job.

- **URL**: `/status/{job_id}`
- **Method**: `GET`
- **URL Parameters**:
  - `job_id` (string, required): Unique ID returned by `/predict` or `/finetune`.
- **Response** (200 OK):

  Pending/Processing:

    ```json
    {
      "status": "PROCESSING",
      "message": "Video processing in progress.",
      "progress": "50%",
      "current_frame": 150,
      "total_frames": 300
    }
    ```

    (or similar for fine-tuning, with `current_epoch`, `total_epochs`, `metrics`)
    Note: When pending/processing, includes `Retry-After: 5` header for polling.

  Completed (Prediction):

    ```json
    {
      "status": "COMPLETED",
      "message": "Object counting completed successfully.",
      "output_video_location": "data/output/counted_20250715123456_my_video.mp4",
      "progress": "100%"
    }
    ```

  Completed (Fine-tuning):

    ```json
    {
      "status": "COMPLETED",
      "message": "Model fine-tuning completed successfully. Use the 'model_name' parameter in /predict to use this model.",
      "fine_tuned_model_path": "data/finetuned_models/maize_plant_detector/weights/best.pt",
      "progress": "100%"
    }
    ```

  Failed:

    ```json
    {
      "status": "FAILED",
      "message": "An error occurred during video processing.",
      "error": "Details of the error message.",
      "progress": "Error"
    }
    ```

  Job ID Not Found (404 Not Found):

    ```json
    {"detail": "Job ID not found."}
    ```

#### Example Usage (using `curl`)

```bash
curl -X GET "http://localhost:8000/status/a1b2c3d4-e5f6-7899-1234-567890abcdef" \
     -H "accept: application/json"
```


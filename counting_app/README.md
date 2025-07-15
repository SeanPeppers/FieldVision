
# Define the markdown content
markdown_content = """# YOLO11 Object Counting API

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
  - [Endpoint](#endpoint)
  - [Example Request](#example-request)
  - [Example Response](#example-response)
- [Model and Configuration](#model-and-configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure


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

## API Usage check available API endponts at 
```
 "http://localhost:8000/docs" 
```
### Endpoint

```http
POST /predict
Example Request

curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/videos/solutions-ci-demo.mp4"
```

Example Response

```
{
  "status": "success",
  "message": "Object counting completed successfully.",
  "output_video_location": "data/output/counted_YYYYMMDDHHMMSS_your_video.mp4"
}
```
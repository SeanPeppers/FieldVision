# Structure from Motion (SfM) Pipeline – FastAPI Application

This FastAPI application provides a pipeline for performing Structure from Motion (SfM) on video inputs using the VGGT model. It allows for asynchronous processing and real-time job status monitoring.

---

## Key Features

- **Asynchronous Processing**: Submit video processing jobs that run in the background, allowing the API to remain responsive.
- **Job Status Tracking**: Monitor the progress and status of individual SfM jobs using a unique Job ID.
- **System Metrics**: An endpoint to check the Docker container's CPU, memory, and I/O usage in real-time.
- **VGGT Integration**: Leverages the VGGT model (1 Billion parameters) for camera pose and 3D point cloud estimation.

---

## API Endpoints

### `/` (GET)

- **Description**: Root endpoint. Returns a welcome message.

---

### `/sfm_predict` (POST)

- **Description**: Initiates an SfM processing job for an uploaded video.
- **Parameters**:
  - `file` (File): The input video file.
  - `output_name` (Form, optional): Base name for output files. Defaults to a timestamp.
  - `sample_frames` (Form, optional): Number of frames to sample from the video. Defaults to 50.
- **Response**: Returns a `job_id` for tracking. The API responds immediately, and processing occurs in the background.

---

### `/sfm_status/{job_id}` (GET)

- **Description**: Retrieves the current status of an SfM job.
- **Parameters**:
  - `job_id` (Path): The unique ID of the job to check.
- **Response**: Returns the job's status, progress, and output file locations (if completed). Includes a `Retry-After` header if still processing.

---

### `/compute_status` (GET)

- **Description**: Provides real-time CPU, memory, disk I/O, and network I/O usage of the Docker container. This endpoint is designed to be non-blocking and always responsive.

---

## Outputs

For each successful `/sfm_predict` job, the following files are generated and stored in the mounted `data/outputs/` directory on your host machine:

- `_intrinsics.json`: Camera intrinsic parameters.
- `_extrinsics.json`: Camera extrinsic parameters (poses) for each sampled frame.
- `_point_cloud.ply`: The reconstructed 3D point cloud in PLY format.

These outputs provide the necessary data for integration with 3D Gaussian Splatting frameworks or other 3D visualization tools.

---

## Limitations

- **VGGT Model Size**: The VGGT model is very large (1 Billion parameters). It will be downloaded from Hugging Face Hub on the first startup and takes a significant amount of time to load.
- **GPU Requirement**: While the application can technically fall back to CPU, running the VGGT model on a CPU is highly unrecommended due to extreme slowness and high memory consumption leading to potential crashes. A CUDA-enabled GPU is essential for practical use.
- **COLMAP Binary Output**: The current implementation does not directly output COLMAP's native binary (.bin) files due to dependency conflicts. It provides equivalent data in JSON and PLY formats.

---

## Docker Methods

### 1. Build the Docker Image

Navigate to the `sfm_app` directory and run:

```bash
docker build -t sfm-pipeline:latest .
```

---

### 2. Run the Docker Container

#### Highly Recommended (with GPU)

To utilize your GPU (essential for VGGT performance), ensure you have the NVIDIA Container Toolkit installed on your host system. Then, run with the `--gpus all` flag:

```bash
docker run -d -p 8000:8000   --gpus all   -v "$(pwd)/sfm_app/data/videos:/app/data/videos"   -v "$(pwd)/sfm_app/data/outputs:/app/data/outputs"   --name sfm-container sfm-pipeline:latest
```

#### CPU-Only (Not Recommended for Production)

If no GPU is available, you can run without the `--gpus all` flag. Be aware that performance will be extremely slow, and memory issues are likely with larger videos or `sample_frames` values.

```bash
docker run -d -p 8000:8000   -v "$(pwd)/sfm_app/data/videos:/app/data/videos"   -v "$(pwd)/sfm_app/data/outputs:/app/data/outputs"   --name sfm-container sfm-pipeline:latest
```

---

### 3. Monitor Container Logs

To view detailed startup logs and job progress (including model loading):

```bash
docker logs -f sfm-container
```

---

## Testing the API

Once the container is running and the VGGT model has finished loading (check docker logs), you can test the API:

1. **Access Swagger UI**: Open your web browser and navigate to `http://localhost:8000/docs#/`.
2. **Upload a video**: Use the `/sfm_predict` endpoint to upload a video.
3. **Check status**: Use the `job_id` returned by `/sfm_predict` to query the `/sfm_status/{job_id}` endpoint.
4. **Monitor compute**: Continuously query `/compute_status` to observe container resource usage.

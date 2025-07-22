# COLMAP CLI SfM API

This FastAPI application provides a wrapper for the COLMAP Command Line Interface (CLI) to perform Structure from Motion (SfM) reconstruction from video inputs. It focuses on generating a sparse 3D model and undistorted images.

## Pipeline Steps Included

- Feature Extraction  
- Exhaustive Matching  
- Sparse Reconstruction (Mapper)  
- Bundle Adjustment  
- Image Undistortion  

> **Note:** Dense reconstruction steps like Stereo Matching, Fusion, and Meshing are excluded for a simplified pipeline.

---

## Setup

### Build Docker Image

Navigate to your `colmap_app` directory and build the Docker image.

**For ARM-based macOS (M1/M2/M3 Macs):**
```bash
docker build --platform=linux/amd64 -t colmap-pipeline:latest .
```

**For AMD64 Linux or Windows (WSL2):**
```bash
docker build -t colmap-pipeline:latest .
```

### Run Docker Container

```bash
docker run -d -p 8000:8000 \
  -v "$(pwd)/data/videos:/app/data/videos" \
  -v "$(pwd)/data/outputs:/app/data/outputs" \
  --name colmap-container colmap-pipeline:latest
```

> Add `--gpus all` to `docker run` if you have a compatible NVIDIA GPU and the NVIDIA Container Toolkit installed for GPU acceleration.

---

## API Usage

Access the API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

### `/colmap_predict` (POST)

Upload a video to initiate SfM processing.

**Parameters:**

- `file`: The video file.  
- `output_name` (optional): Base name for output files.  
- `sample_frames` (optional, default: 50): Number of frames to sample from the video.  
- `use_gpu` (optional, default: True): Whether to use GPU for COLMAP steps. Set to False to force CPU.  
- `max_image_size` (optional, default: 4000): Max image size for undistortion.

**Example `curl`:**
```bash
curl -X POST "http://localhost:8000/colmap_predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/video.mp4" \
     -F "use_gpu=false"
```

---

### `/sfm_status/{job_id}` (GET)

Check the status of a running or completed job.

**Example `curl`:**
```bash
curl -X GET "http://localhost:8000/sfm_status/YOUR_JOB_ID"
```

---

### `/compute_status` (GET)

Get container resource usage (CPU, memory, disk, network).

---

## Output Files

All generated COLMAP output files, including the sparse 3D model, undistorted images, and intermediate COLMAP workspace files, will be saved to the `data/outputs` directory on your host machine. This is due to the volume mount:

```bash
-v "$(pwd)/data/outputs:/app/data/outputs"
```

Input video files are temporarily stored in `data/videos` on your host machine before processing and are automatically cleaned up after the job completes.

For the exact paths to the generated sparse model and undistorted images, refer to the `output_files` field in the response from the `/sfm_status/{job_id}` endpoint when a job is `COMPLETED`. The `colmap_workspace` path within this output will point to the full COLMAP project directory for detailed inspection.

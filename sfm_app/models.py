"""
This module defines the SfMProcessor class responsible for handling
Structure from Motion (SfM) processing using the VGGT model directly.
It includes methods for video frame extraction, invoking the VGGT model
for camera pose and 3D point estimation, and saving outputs.
"""

import os
import subprocess
import asyncio
import cv2
import json
import shutil
import numpy as np
from PIL import Image # Used for image saving
import logging # Import logging module

# Configure logging for this module
logger = logging.getLogger(__name__)
# Basic configuration (can be overridden by main app's configuration)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Import necessary VGGT modules
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

class SfMProcessor:
    def __init__(self, vggt_repo_path: str):
        """
        Initializes the SfMProcessor.

        Args:
            vggt_repo_path (str): The path to the cloned VGGT repository within the container.
        """
        # Assign vggt_repo_path first to avoid AttributeError
        self.vggt_repo_path = vggt_repo_path
        logger.info(f"SfMProcessor: Initializing with VGGT path: {self.vggt_repo_path}")

        # Determine device (CUDA if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"SfMProcessor: Using device: {self.device}")

        # Determine dtype (bfloat16 for Ampere GPUs+, float16 otherwise)
        self.dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        logger.info(f"SfMProcessor: Using dtype: {self.dtype}")

        # Initialize the VGGT model and load pretrained weights.
        # This will automatically download the model weights the first time it's run.
        try:
            logger.info("SfMProcessor: Attempting to load VGGT model from Hugging Face Hub (facebook/VGGT-1B)... This may take a while.")
            self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
            self.model.eval() # Set model to evaluation mode
            logger.info("SfMProcessor: VGGT model loaded successfully.")
        except Exception as e:
            logger.error(f"SfMProcessor ERROR: Failed to load VGGT model: {e}", exc_info=True)
            # Fallback or raise error if model cannot be loaded
            raise RuntimeError(f"Failed to load VGGT model: {e}")


    async def _extract_frames(self, video_path: str, output_dir: str, job_id: str, sample_frames: int) -> str:
        """
        Extracts frames from a video and saves them as images.
        This operation is blocking, so it's run in a separate thread.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory where extracted frames will be saved.
            job_id (str): The ID of the current job for logging.
            sample_frames (int): The number of frames to sample from the video.

        Returns:
            str: Path to the directory containing the extracted images.
        """
        frames_output_path = os.path.join(output_dir, "images")
        os.makedirs(frames_output_path, exist_ok=True)

        logger.info(f"Job {job_id}: Extracting frames from {video_path} to {frames_output_path} with {sample_frames} samples.")

        # Use asyncio.to_thread for blocking OpenCV operations
        def _blocking_frame_extraction():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_video_frames == 0:
                raise ValueError("Video contains no frames.")

            # Calculate frame indices to sample
            num_frames_to_sample = min(sample_frames, total_video_frames)
            
            if num_frames_to_sample > 1:
                frame_indices = np.linspace(0, total_video_frames - 1, num_frames_to_sample, dtype=int)
            else: # If sampling 1 frame, just take the first
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

        image_folder, saved_frame_count = await asyncio.to_thread(_blocking_frame_extraction)
        if saved_frame_count == 0:
            raise RuntimeError("No frames were extracted from the video.")
        
        logger.info(f"Job {job_id}: Extracted {saved_frame_count} sampled frames.")
        return image_folder

    async def _run_vggt_colmap_demo(self, image_folder: str, vggt_scene_dir: str, job_id: str):
        """
        Runs the VGGT demo_colmap.py script as a subprocess.
        This operation is blocking, so it's run in a separate thread.

        Args:
            image_folder (str): Path to the folder containing input images (e.g., /path/to/frames/images).
            vggt_scene_dir (str): The base directory that VGGT's demo_colmap.py will use as --scene_dir.
                                  VGGT will create 'images' and 'sparse' folders inside it.
            job_id (str): The ID of the current job for logging.
        """
        # Ensure the scene_dir exists for VGGT
        os.makedirs(vggt_scene_dir, exist_ok=True)

        # NOTE: The previous logic for temp_frames_dir and vggt_scene_dir was a bit confusing.
        # Let's clarify:
        # _extract_frames saves to `temp_scene_dir/images` (e.g., `/tmp/vggt_scene_XYZ/images`)
        # `demo_colmap.py` expects `--scene_dir` to be the parent of `images` (e.g., `/tmp/vggt_scene_XYZ`)
        # So, the `image_folder` passed to this method should be `temp_scene_dir` (which contains the `images` subfolder).
        
        # We already ensured `temp_scene_dir` contains `images` in `_extract_frames`.
        # So, `scene_dir` for VGGT is simply `image_folder` (e.g., /tmp/vggt_scene_XYZ)
        scene_dir_for_vggt = image_folder # This is the parent directory that holds the 'images' folder

        command = [
            "python",
            os.path.join(self.vggt_repo_path, "demo_colmap.py"),
            f"--scene_dir={scene_dir_for_vggt}"
            # Add --use_ba if bundle adjustment is desired later
            # Add --max_query_pts, --query_frame_num for BA optimization
        ]
        
        logger.info(f"Job {job_id}: Running VGGT demo_colmap.py command: {' '.join(command)}")
        
        # Use asyncio.to_thread for blocking subprocess execution
        def _blocking_subprocess_run():
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            return process.returncode, stdout, stderr

        returncode, stdout, stderr = await asyncio.to_thread(_blocking_subprocess_run)

        if returncode != 0:
            logger.error(f"Job {job_id}: VGGT demo_colmap.py stdout:\n{stdout.decode()}")
            logger.error(f"Job {job_id}: VGGT demo_colmap.py stderr:\n{stderr.decode()}")
            raise RuntimeError(f"VGGT processing failed with exit code {returncode}. Error: {stderr.decode()}")
        
        logger.info(f"Job {job_id}: VGGT demo_colmap.py completed successfully.")
        logger.info(f"Job {job_id}: VGGT stdout (partial):\n{stdout.decode()[-1000:]}") # Print last 1000 chars
        logger.info(f"Job {job_id}: VGGT stderr (partial):\n{stderr.decode()[-1000:]}") # Print last 1000 chars


    async def process_video(self, video_path: str, output_base_path: str, progress_callback: callable, sample_frames: int) -> dict:
        """
        Orchestrates the SfM processing for a given video using VGGT modules.

        Args:
            video_path (str): Path to the input video.
            output_base_path (str): Base path for saving output files (e.g., /data/outputs/job_name).
            progress_callback (callable): A function to report progress updates.
            sample_frames (int): The number of frames to sample from the video for processing.

        Returns:
            dict: Paths to the generated output files (intrinsics, extrinsics, point_cloud).
        """
        job_id = os.path.basename(output_base_path) # Use output_base_path as job_id for internal logging
        
        # temp_scene_dir will be the parent directory for VGGT's 'images' and 'sparse' outputs
        temp_scene_dir = os.path.join("/tmp", f"vggt_scene_{job_id}")
        
        # Output paths for the final JSON/PLY files
        final_output_files = {
            "intrinsics": f"{output_base_path}_intrinsics.json",
            "extrinsics": f"{output_base_path}_extrinsics.json",
            "point_cloud": f"{output_base_path}_point_cloud.ply"
        }

        try:
            # Stage 1: Frame Extraction (with sampling)
            progress_callback(10, "Extracting Frames")
            logger.info(f"Job {job_id}: Starting frame extraction with sampling {sample_frames} frames.")
            
            # Pass sample_frames to _extract_frames
            image_folder = await self._extract_frames(video_path, temp_scene_dir, job_id, sample_frames)
            
            if not os.path.exists(image_folder) or not os.listdir(image_folder):
                raise RuntimeError("No frames were extracted from the video or image folder is empty.")
            
            logger.info(f"Job {job_id}: Extracted sampled frames to {image_folder}.")
            progress_callback(30, "Frames Extracted")

            # Stage 2: VGGT Inference
            progress_callback(40, "Running VGGT Inference")
            logger.info(f"Job {job_id}: Starting VGGT inference on extracted frames.")
            
            # Get list of image paths
            image_names = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            if not image_names:
                raise ValueError("No images found for VGGT processing after sampling.")

            # Load and preprocess images for VGGT
            logger.info(f"Job {job_id}: Loading and preprocessing {len(image_names)} images for VGGT.")
            images_tensor = await asyncio.to_thread(load_and_preprocess_images, image_names)
            images_tensor = images_tensor.to(self.device)
            
            images_tensor = images_tensor[None] # Add batch dimension: (1, N, C, H, W)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    logger.info(f"Job {job_id}: Running VGGT model forward pass.")
                    predictions = self.model(images_tensor)

                    pose_enc = predictions['pose_enc']
                    depth_map = predictions['depth_map']

                    logger.info(f"Job {job_id}: Converting pose encoding to extrinsic/intrinsic matrices.")
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])

                    logger.info(f"Job {job_id}: Unprojecting depth map to point cloud.")
                    point_cloud_tensor = unproject_depth_map_to_point_map(
                        depth_map.squeeze(0),
                        extrinsic.squeeze(0),
                        intrinsic.squeeze(0)
                    )
                    point_cloud_data = point_cloud_tensor.reshape(-1, 3).cpu().numpy()

            logger.info(f"Job {job_id}: VGGT Inference Complete.")
            progress_callback(70, "VGGT Inference Complete")

            # Stage 3: Save Outputs
            progress_callback(80, "Saving Outputs")
            logger.info(f"Job {job_id}: Saving intrinsic, extrinsic, and point cloud data.")

            # Save Intrinsics (from the first camera, assuming it's consistent)
            intrinsics_to_save = intrinsic.squeeze(0)[0].cpu().numpy().tolist()
            with open(final_output_files["intrinsics"], "w") as f:
                json.dump({"intrinsic_matrix": intrinsics_to_save}, f, indent=4)

            # Save Extrinsics (poses for each frame)
            extrinsics_to_save = {}
            for i, img_name in enumerate(image_names):
                frame_id = os.path.basename(img_name).split('.')[0] # e.g., "000000"
                extrinsics_to_save[frame_id] = extrinsic.squeeze(0)[i].cpu().numpy().tolist()
            with open(final_output_files["extrinsics"], "w") as f:
                json.dump(extrinsics_to_save, f, indent=4)
            
            # Save Point Cloud to PLY
            with open(final_output_files["point_cloud"], "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(point_cloud_data)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                for p in point_cloud_data:
                    if isinstance(p, list) and len(p) == 3: # Ensure it's a valid point
                        f.write(f"{p[0]} {p[1]} {p[2]}\n")
                    else:
                        logger.warning(f"Job {job_id}: Skipping invalid point data: {p}")


            progress_callback(90, "Outputs Saved")
            logger.info(f"Job {job_id}: Outputs saved successfully.")
            
            return final_output_files

        except Exception as e:
            logger.error(f"Job {job_id}: Error during SfM processing: {e}", exc_info=True)
            raise # Re-raise the exception to be caught by the calling background task
        finally:
            # Clean up temporary directories
            if os.path.exists(temp_scene_dir):
                logger.info(f"Job {job_id}: Cleaning up temporary VGGT scene directory: {temp_scene_dir}")
                shutil.rmtree(temp_scene_dir)


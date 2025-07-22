# colmap_app/colmap_processor.py
import os
import subprocess
import asyncio
import logging
import shutil
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ColmapProcessor:
    def __init__(self):
        logger.info("ColmapProcessor: Initializing. Ensuring COLMAP CLI is available.")
        # Check if colmap is in PATH
        if shutil.which("colmap") is None:
            logger.error("COLMAP executable not found in PATH. Please ensure COLMAP is installed and accessible.")
            raise RuntimeError("COLMAP executable not found.")
        logger.info("ColmapProcessor: COLMAP CLI detected.")

    async def _run_colmap_command(self, command: list, job_id: str, step_name: str):
        """Helper to run a COLMAP command in a separate thread."""
        logger.info(f"Job {job_id}: Running COLMAP {step_name} command: {' '.join(command)}")

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
            logger.error(f"Job {job_id}: COLMAP {step_name} stdout:\n{stdout.decode()}")
            logger.error(f"Job {job_id}: COLMAP {step_name} stderr:\n{stderr.decode()}")
            raise RuntimeError(f"COLMAP {step_name} failed with exit code {returncode}. Error: {stderr.decode()}")
        
        logger.info(f"Job {job_id}: COLMAP {step_name} completed successfully.")
        logger.debug(f"Job {job_id}: COLMAP {step_name} stdout (partial):\n{stdout.decode()[-1000:]}")
        logger.debug(f"Job {job_id}: COLMAP {step_name} stderr (partial):\n{stderr.decode()[-1000:]}")

    async def process_images(self, image_folder: str, output_base_path: str, job_id: str, use_gpu: bool = True, max_image_size: int = 4000):
        """
        Processes a folder of images using COLMAP CLI for SfM reconstruction,
        including feature extraction, matching, mapping, bundle adjustment, and image undistortion.
        Dense reconstruction steps (stereo matching, fusion, meshing) are excluded.

        Args:
            image_folder (str): Path to the folder containing input images.
            output_base_path (str): Base path for saving all COLMAP outputs.
            job_id (str): Unique identifier for the current job.
            use_gpu (bool): Whether to use GPU for COLMAP steps.
            max_image_size (int): Max image size for undistortion (to prevent downsampling).
        Returns:
            dict: Paths to the generated output files (sparse model, undistorted images).
        """
        logger.info(f"Job {job_id}: Starting COLMAP reconstruction for images in {image_folder}")

        # Define output paths
        colmap_output_dir = os.path.join(output_base_path, "colmap_reconstruction")
        os.makedirs(colmap_output_dir, exist_ok=True)

        database_path = os.path.join(colmap_output_dir, "database.db")
        sparse_output_path = os.path.join(colmap_output_dir, "sparse")
        dense_output_path = os.path.join(colmap_output_dir, "dense") # Still create for undistorted images
        
        os.makedirs(sparse_output_path, exist_ok=True)
        os.makedirs(dense_output_path, exist_ok=True)

        # Determine GPU flag value (0 for CPU, 1 for GPU)
        gpu_flag_value = "1" if use_gpu else "0"

        # 1. Feature Extraction
        feature_extractor_cmd = [
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_folder,
            "--ImageReader.single_camera", "1", # Use a single camera model
            "--SiftExtraction.use_gpu", gpu_flag_value # Explicitly set GPU usage
        ]
        await self._run_colmap_command(feature_extractor_cmd, job_id, "Feature Extraction")

        # 2. Exhaustive Matching
        exhaustive_matcher_cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", database_path,
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.max_num_matches", "32768",
            "--SiftMatching.use_gpu", gpu_flag_value, # Explicitly set GPU usage,
            "--SiftMatching.num_threads", "4"
        ]
        await self._run_colmap_command(exhaustive_matcher_cmd, job_id, "Exhaustive Matching")

        # 3. Sparse Reconstruction (Mapper)
        mapper_cmd = [
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", image_folder,
            "--output_path", sparse_output_path,
            "--Mapper.init_min_num_inliers", "50",
            "--Mapper.filter_max_reproj_error", "4.0",
            "--Mapper.multiple_models", "0"
        ]
        await self._run_colmap_command(mapper_cmd, job_id, "Sparse Reconstruction (Mapper)")

        # Check if sparse model was created (colmap mapper creates a '0' subdirectory)
        sparse_model_input_path = os.path.join(sparse_output_path, "0")
        if not os.path.exists(sparse_model_input_path) or not os.listdir(sparse_model_input_path):
            raise RuntimeError(f"Job {job_id}: COLMAP Sparse Reconstruction failed to produce a model in {sparse_model_input_path}.")
        logger.info(f"Job {job_id}: Sparse model generated at {sparse_model_input_path}")

        # 4. Bundle Adjustment (Global)
        # This step refines the sparse reconstruction
        bundle_adjuster_cmd = [
            "colmap", "bundle_adjuster",
            "--input_path", sparse_model_input_path,
            "--output_path", sparse_model_input_path, # Overwrite the existing model with refined one
            "--BundleAdjustment.refine_principal_point", "true",
            "--BundleAdjustment.refine_focal_length", "true",
            "--BundleAdjustment.refine_extra_params", "true",
            "--BundleAdjustment.max_num_iterations", "100"
        ]
        await self._run_colmap_command(bundle_adjuster_cmd, job_id, "Bundle Adjustment")
        logger.info(f"Job {job_id}: Sparse model refined by Bundle Adjustment.")

        # 5. Undistort Images
        # Use the refined sparse model for undistortion
        current_model_path_for_undistort = sparse_model_input_path
        image_undistorter_cmd = [
            "colmap", "image_undistorter",
            "--image_path", image_folder,
            "--input_path", current_model_path_for_undistort,
            "--output_path", dense_output_path,
            "--output_type", "COLMAP", # This creates a workspace with undistorted images and a copy of the sparse model
            "--max_image_size", str(max_image_size) # Prevent downsampling
        ]
        await self._run_colmap_command(image_undistorter_cmd, job_id, "Image Undistortion")
        
        # Check if undistorted images and dense workspace are created
        if not os.path.exists(os.path.join(dense_output_path, "images")) or \
           not os.path.exists(os.path.join(dense_output_path, "sparse")): # Undistorter also outputs sparse model
            raise RuntimeError(f"Job {job_id}: COLMAP Image Undistortion failed to produce expected output in {dense_output_path}.")
        logger.info(f"Job {job_id}: Images undistorted to {dense_output_path}/images")

        # Dense reconstruction steps (Stereo Matching, Stereo Fusion, Meshing) are excluded as requested.
        logger.info(f"Job {job_id}: Dense reconstruction steps (Stereo Matching, Fusion, Meshing) skipped as requested.")

        logger.info(f"Job {job_id}: COLMAP reconstruction complete.")

        return {
            "sparse_model_path": sparse_model_input_path, # Path to the '0' directory with refined model
            "undistorted_images_path": os.path.join(dense_output_path, "images"), # Path to the undistorted images
            "colmap_workspace": colmap_output_dir # Return the full workspace path for inspection
        }

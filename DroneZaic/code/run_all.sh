#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

APP_DIR="/app"
OUTPUTS_DIR="outputs"
CALIBRATED_FRAMES_DIR="$OUTPUTS_DIR/calibrated_frames"
HOMOGRAPHY_RESULTS_DIR="$OUTPUTS_DIR/homography_results"
HOMOGRAPHY_MATRICES_FILE="$HOMOGRAPHY_RESULTS_DIR/homography_matrices/H_surf.csv" # Assuming SURF for this example

# Define the directories to be cleared
MINI_MOSAICS_DIR="$OUTPUTS_DIR/surf_mini_mosaics"
MINI_PARTITION_DIR="$OUTPUTS_DIR/surf_mini_partition"
GLOBAL_MOSAIC_TILED_DIR="$OUTPUTS_DIR/surf_global_mosaic_tiled" # Also good to clear this for a clean run

echo "Starting Maizaic pipeline with conditional steps..."

# --- Clear specified output directories for a clean run ---
echo "--- Clearing previous mini-mosaic and partition outputs ---"
if [ -d "$MINI_MOSAICS_DIR" ]; then
    echo "Removing $MINI_MOSAICS_DIR..."
    rm -rf "$MINI_MOSAICS_DIR"
fi
if [ -d "$MINI_PARTITION_DIR" ]; then
    echo "Removing $MINI_PARTITION_DIR..."
    rm -rf "$MINI_PARTITION_DIR"
fi
if [ -d "$GLOBAL_MOSAIC_TILED_DIR" ]; then
    echo "Removing $GLOBAL_MOSAIC_TILED_DIR..."
    rm -rf "$GLOBAL_MOSAIC_TILED_DIR"
fi
echo "Directory cleanup complete."

# Ensure the parent output directories exist before any steps create files in them
mkdir -p "$CALIBRATED_FRAMES_DIR"
mkdir -p "$HOMOGRAPHY_RESULTS_DIR/homography_matrices" # Ensure homography matrices folder is ready

# --- Step 1: Calibration ---
# This step might need to re-run if calibrated_frames are removed (e.g., if you add them to the clear list)
if [ ! -d "$CALIBRATED_FRAMES_DIR" ] || [ -z "$(ls -A "$CALIBRATED_FRAMES_DIR" 2>/dev/null)" ]; then
    echo "--- Calibrated frames not found or empty. Running Calibration. ---"
    cd "$APP_DIR"
    python code/calibration.py \
    -image_path "$OUTPUTS_DIR/extracted_frames/raw" \
    -save_path "$CALIBRATED_FRAMES_DIR"
    echo "Calibration complete."
else
    echo "--- Calibrated frames already exist. Skipping Calibration. ---"
fi


# --- Step 2: Homography Estimation (SURF) ---
# Check for a specific expected output file as well
if [ ! -f "$HOMOGRAPHY_MATRICES_FILE" ]; then
    echo "--- Homography matrix file not found. Running Homography Estimation (SURF). ---"
    cd "$APP_DIR" # Ensure we are in /app if previous step changed directory
    python code/surf/surf_homography_estimation.py \
    -image_path "$CALIBRATED_FRAMES_DIR" \
    -save_path "$HOMOGRAPHY_RESULTS_DIR" \
    -scale 1
    echo "Homography Estimation complete."
else
    echo "--- Homography matrices already exist. Skipping Homography Estimation. ---"
fi


# --- Step 3: Maizaic Run (Mini-mosaics and Global Mosaic) ---
echo "--- Running Maizaic Global Mosaic Assembly ---"
cd "$APP_DIR" # Ensure we are in /app
./code/maizaic_run.sh \
-p "$OUTPUTS_DIR" \
-h surf \
-d false

echo "Maizaic Global Mosaic Assembly complete."

echo "All relevant steps finished successfully!"
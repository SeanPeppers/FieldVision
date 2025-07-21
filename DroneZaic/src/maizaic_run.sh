#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Starting Full MaiZaic Mini-Mosaic Generation Pipeline ---"

# --- CONFIGURATION VARIABLES ---
# Read these from environment variables provided by the API
RAW_VIDEO_PATH="${RAW_VIDEO_PATH:-/app/input_data/default_video.MOV}" # Default if not set
SRT_FILE_PATH="${SRT_FILE_PATH:-}" # Default to empty if not set

# Adjust gimbal calibration angles
GIMBAL_CALIB_X="${GIMBAL_CALIB_X:-0.0}"
GIMBAL_CALIB_Y="${GIMBAL_CALIB_Y:-0.0}"
GIMBAL_CALIB_Z="${GIMBAL_CALIB_Z:-0.0}"

# BASE OUTPUT DIRECTORY
BASE_OUTPUTS_DIR="${BASE_OUTPUTS_DIR:-/app/outputs}" # Default to /app/outputs

# Specific subdirectories for each stage
DYNAMIC_SAMPLING_OUTPUT_DIR="$BASE_OUTPUTS_DIR/extracted_frames"
CALIBRATION_OUTPUT_DIR="$BASE_OUTPUTS_DIR/calibrated_frames"
HOMOGRAPHY_RESULTS_DIR="$BASE_OUTPUTS_DIR/homography_results"
HM_METHOD="${HM_METHOD:-asift}" # Default to asift

MINI_PARTITION_DIR="$BASE_OUTPUTS_DIR/${HM_METHOD}_mini_partition"
MINI_MOSAICS_DIR="$BASE_OUTPUTS_DIR/${HM_METHOD}_mini_mosaics"
SUPERGLUE_MOSAICS_DIR="$BASE_OUTPUTS_DIR/glue_mini_mosaics"

# Other parameters
DYNAMIC_SAMPLING_FNAME="${DYNAMIC_SAMPLING_FNAME:-DJI_processed_frames}"
DYNAMIC_SAMPLING_FORMAT="${DYNAMIC_SAMPLING_FORMAT:-jpg}"
DYNAMIC_SAMPLING_FPS="${DYNAMIC_SAMPLING_FPS:-0.5}"
DYNAMIC_SAMPLING_WIN_SIZE="${DYNAMIC_SAMPLING_WIN_SIZE:-50}"
DYNAMIC_SAMPLING_START_NUMBER="${DYNAMIC_SAMPLING_START_NUMBER:-1}"
DYNAMIC_SAMPLING_SS="${DYNAMIC_SAMPLING_SS:-0}"
DYNAMIC_SAMPLING_SCALE="${DYNAMIC_SAMPLING_SCALE:-1}"
MINI_MOSAIC_STITCHER_SCALE="${MINI_MOSAIC_STITCHER_SCALE:-4}"

# --- Environment variable check for debugging ---
echo "--- Pipeline Configuration (from Environment) ---"
echo "RAW_VIDEO_PATH: $RAW_VIDEO_PATH"
echo "SRT_FILE_PATH: $SRT_FILE_PATH"
echo "GIMBAL_CALIB_X: $GIMBAL_CALIB_X"
# ... print other key variables for debugging
echo "--- End Configuration ---"

# --- DIRECTORY MANAGEMENT ---
echo "--- Ensuring output directories exist and clearing specific ones ---"
mkdir -p "$DYNAMIC_SAMPLING_OUTPUT_DIR"
mkdir -p "$CALIBRATION_OUTPUT_DIR"
mkdir -p "$HOMOGRAPHY_RESULTS_DIR/homography_matrices"
mkdir -p "$MINI_PARTITION_DIR"
mkdir -p "$MINI_MOSAICS_DIR"
mkdir -p "$SUPERGLUE_MOSAICS_DIR"

# Clear specific directories for a clean run, as per previous discussions
echo "Clearing previous dynamic sampling outputs..."
rm -rf "$DYNAMIC_SAMPLING_OUTPUT_DIR"/* || true
mkdir -p "$DYNAMIC_SAMPLING_OUTPUT_DIR"

# Clear calibrated frames as well
echo "Clearing previous calibrated frames..."
rm -rf "$CALIBRATION_OUTPUT_DIR"/* || true
mkdir -p "$CALIBRATION_OUTPUT_DIR"

# run_mini.sh already handles clearing MINI_PARTITION_DIR and MINI_MOSAICS_DIR

echo "Clearing previous SuperGlue mini mosaics..."
rm -rf "$SUPERGLUE_MOSAICS_DIR"/* || true
mkdir -p "$SUPERGLUE_MOSAICS_DIR"

echo "All necessary directories prepared."

start_time_full_pipeline=$(date +%s)

# --- PHASE 1: DYNAMIC SAMPLING (Extract frames with movement awareness and GPS) ---
echo "--- Running Dynamic Sampling ---"
DYNAMIC_SAMPLING_LOG_FILE="$BASE_OUTPUTS_DIR/dynamic_sampling_log.txt"

dynamic_sampling_cmd="python dynamic_sampling.py \
    -video \"$RAW_VIDEO_PATH\" \
    -save_path \"$DYNAMIC_SAMPLING_OUTPUT_DIR\" \
    -fname \"$DYNAMIC_SAMPLING_FNAME\" \
    -format \"$DYNAMIC_SAMPLING_FORMAT\" \
    -fps $DYNAMIC_SAMPLING_FPS \
    -win $DYNAMIC_SAMPLING_WIN_SIZE \
    -start_number $DYNAMIC_SAMPLING_START_NUMBER \
    -ss $DYNAMIC_SAMPLING_SS \
    -scale $DYNAMIC_SAMPLING_SCALE"

if [ -n "$SRT_FILE_PATH" ]; then
    dynamic_sampling_cmd="$dynamic_sampling_cmd -srt \"$SRT_FILE_PATH\""
fi

eval "$dynamic_sampling_cmd" | tee "$DYNAMIC_SAMPLING_LOG_FILE"
if [ $? -ne 0 ]; then echo "Error: dynamic_sampling.py failed."; exit 1; fi
echo "Dynamic Sampling complete. Logs in $DYNAMIC_SAMPLING_LOG_FILE"

EXTRACTED_RAW_FRAMES_DIR="$DYNAMIC_SAMPLING_OUTPUT_DIR/raw"
if [ ! -d "$EXTRACTED_RAW_FRAMES_DIR" ] || [ -z "$(ls -A "$EXTRACTED_RAW_FRAMES_DIR" 2>/dev/null)" ]; then
    echo "Error: Dynamic sampling output directory '$EXTRACTED_RAW_FRAMES_DIR' is empty or not found. Cannot proceed."
    exit 1
fi
echo "Extracted frames found in $EXTRACTED_RAW_FRAMES_DIR."

# --- PHASE 2: CALIBRATION (Undistort and apply gimbal correction) ---
echo "--- Running Calibration ---"
CALIBRATION_LOG_FILE="$BASE_OUTPUTS_DIR/calibration_log.txt"

calibration_cmd="python calibration.py \
    -image_path \"$EXTRACTED_RAW_FRAMES_DIR\" \
    -save_path \"$CALIBRATION_OUTPUT_DIR\" \
    -xxx $GIMBAL_CALIB_X \
    -yyy $GIMBAL_CALIB_Y \
    -zzz $GIMBAL_CALIB_Z"

eval "$calibration_cmd" | tee "$CALIBRATION_LOG_FILE"
if [ $? -ne 0 ]; then echo "Error: calibration.py failed."; exit 1; fi
echo "Calibration complete. Logs in $CALIBRATION_LOG_FILE"

if [ ! -d "$CALIBRATION_OUTPUT_DIR" ] || [ -z "$(ls -A "$CALIBRATION_OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "Error: Calibrated frames directory '$CALIBRATION_OUTPUT_DIR' is empty or not found. Cannot proceed."
    exit 1
fi
echo "Calibrated frames found in $CALIBRATION_OUTPUT_DIR."

# --- PHASE 3: HOMOGRAPHY ESTIMATION (using ASIFT/SIFT) ---
echo "--- IMPORTANT: Placeholder for Homography Estimation ---"
echo "You need to run your homography estimation script here."
echo "It should use images from: $CALIBRATION_OUTPUT_DIR"
echo "And output H_${HM_METHOD}.csv to: $HOMOGRAPHY_RESULTS_DIR/homography_matrices/"
echo "For example: python homography_estimator.py -image_path \"$CALIBRATION_OUTPUT_DIR\" -save_path \"$HOMOGRAPHY_RESULTS_DIR\" -method \"$HM_METHOD\""
# This is where your actual homography estimation script would go.
# Example command if you had 'homography_estimator.py':
# python homography_estimator.py \
#     -image_path "$CALIBRATION_OUTPUT_DIR" \
#     -save_path "$HOMOGRAPHY_RESULTS_DIR" \
#     -method "$HM_METHOD"
# if [ $? -ne 0 ]; then echo "Error: Homography estimation failed."; exit 1; fi
echo "Assuming homography matrix H_${HM_METHOD}.csv exists at $HOMOGRAPHY_RESULTS_DIR/homography_matrices/"
sleep 2 # Just a pause for the user to read the message

# --- PHASE 4: MINI-MOSAIC PARTITIONING AND STITCHING (Orchestrated by run_mini.sh) ---
echo "--- Running Mini-Mosaic Generation (via run_mini.sh) ---"
MINI_MOSAIC_LOG_FILE="$BASE_OUTPUTS_DIR/mini_mosaic_log.txt"

# Setting environment variables for run_mini.sh if it needs them
export CONTAINER_OUTPUTS_DIR="$BASE_OUTPUTS_DIR" # Pass this so run_mini.sh uses the correct base
export HM_METHOD="$HM_METHOD" # Pass the HM_METHOD to run_mini.sh

pushd /app/code # Assuming run_mini.sh is in /app/code
./run_mini.sh | tee "$MINI_MOSAIC_LOG_FILE"
POP_STATUS=$?
popd
if [ $POP_STATUS -ne 0 ]; then echo "Error: run_mini.sh failed."; exit 1; fi
echo "Mini-Mosaic Generation complete. Logs in $MINI_MOSAIC_LOG_FILE"

# --- PHASE 5: SUPERGLUE PROCESSING (On partitioned images) ---
echo "--- Running SuperGlue Mini-Mosaic Processing ---"
SUPERGLUE_LOG_FILE="$BASE_OUTPUTS_DIR/superglue_log.txt"

# super_glue_mini.py assumes BASE_INPUT_DIR="/app/outputs/asift_mini_partition"
# and GLOBAL_OUTPUT_DIR="/app/outputs/glue_mini_mosaics"
# We need to ensure these paths are correct.
# In super_glue_mini.py:
# BASE_INPUT_DIR = "/app/outputs/asift_mini_partition"
# GLOBAL_OUTPUT_DIR = "/app/outputs/glue_mini_mosaics"
# These should ideally be constructed using BASE_OUTPUTS_DIR and HM_METHOD to be dynamic.
# For now, if HM_METHOD can change, you might need to pass it to super_glue_mini.py
# or modify super_glue_mini.py to be more flexible with its input/output paths.

# For this example, assuming HM_METHOD 'asift' is primarily used.
# If HM_METHOD changes, super_glue_mini.py's hardcoded paths for 'asift_mini_partition'
# and 'glue_mini_mosaics' would need to be updated.
# To make super_glue_mini.py flexible, you'd add argparse arguments to it for base directories.

pushd /app/code
python super_glue_mini.py | tee "$SUPERGLUE_LOG_FILE"
POP_STATUS=$?
popd
if [ $POP_STATUS -ne 0 ]; then echo "Error: super_glue_mini.py failed."; exit 1; fi
echo "SuperGlue Processing complete. Logs in $SUPERGLUE_LOG_FILE"

end_time_full_pipeline=$(date +%s)
elapsed_full_pipeline=$(( end_time_full_pipeline - start_time_full_pipeline ))

echo "--- Full MaiZaic Mini-Mosaic Pipeline Finished ---"
echo "Total Elapsed time for full pipeline: $elapsed_full_pipeline seconds"
echo "Check final SuperGlue results in: $SUPERGLUE_MOSAICS_DIR"
echo "Pipeline completed successfully!"
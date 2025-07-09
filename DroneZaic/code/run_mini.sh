#!/bin/bash
# This script is designed to run ONLY the mini-mosaic generation phases.
# It assumes prior steps (frame extraction, calibration, homography estimation)
# have been successfully completed and their outputs are present.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Define Paths and Parameters ---
APP_DIR="/app"
CONTAINER_OUTPUTS_DIR="/app/outputs" # This will be mounted from host

HM_METHOD="surf" # Hardcoding homography method to 'surf' as per your typical use
MODE_DUPLICATE="true" # Hardcoding mode_duplicate to 'false' as per your typical use

# Define paths for internal use within the container
DYNAMIC_SAMPLING_FRAMES_DIR="$CONTAINER_OUTPUTS_DIR/extracted_frames"
ANGLE_CSV_FILE="$DYNAMIC_SAMPLING_FRAMES_DIR/DJI_0604_frames_angle_diff.csv"
CALIBRATED_FRAMES_DIR="$CONTAINER_OUTPUTS_DIR/calibrated_frames"
HOMOGRAPHY_RESULTS_DIR="$CONTAINER_OUTPUTS_DIR/homography_results"
HOMOGRAPHY_MATRICES_FILE="$HOMOGRAPHY_RESULTS_DIR/homography_matrices/H_${HM_METHOD}.csv"

# Output directories for this script's phases
MINI_PARTITION_DIR="$CONTAINER_OUTPUTS_DIR/${HM_METHOD}_mini_partition"
MINI_MOSAICS_DIR="$CONTAINER_OUTPUTS_DIR/${HM_METHOD}_mini_mosaics"

echo "--- Starting ONLY Mini-Mosaic Generation ---"

# --- Clear previous mini-mosaic and partition outputs for a clean run ---
# RE-ENABLED: User requested to clear these folders for a clean mini-mosaic run.
echo "--- Clearing previous mini-mosaic and partition outputs ---"
if [ -d "$MINI_PARTITION_DIR" ]; then
    echo "Removing $MINI_PARTITION_DIR..."
    rm -rf "$MINI_PARTITION_DIR"
fi
if [ -d "$MINI_MOSAICS_DIR" ]; then
    echo "Removing $MINI_MOSAICS_DIR..."
    rm -rf "$MINI_MOSAICS_DIR"
fi
echo "Directory cleanup complete."

# Ensure output subdirectories exist (will create if they don't, won't error if they do)
mkdir -p "$MINI_PARTITION_DIR"
mkdir -p "$MINI_MOSAICS_DIR"
echo "Ensured output directories are ready."

start_time=$(date +%s)

# --- VALIDATE REQUIRED INPUTS FOR MINI-MOSAIC GENERATION ---
# These are inputs to the split_for_mini.py and stitcher.py scripts.
# The CALIBRATED_FRAMES_DIR is NOT cleared by this script, as per user's request.
if [ ! -f "$ANGLE_CSV_FILE" ]; then
    echo "Error: Angle CSV file '$ANGLE_CSV_FILE' not found. Please ensure it's generated in your mounted outputs/extracted_frames."
    exit 1
fi
# The HOMOGRAPHY_MATRICES_FILE is still needed by split_for_mini.py, even if not by stitcher.py
if [ ! -f "$HOMOGRAPHY_MATRICES_FILE" ]; then
    echo "Error: Homography matrix file '$HOMOGRAPHY_MATRICES_FILE' not found. Please ensure it's generated in your mounted outputs/homography_results."
    exit 1
fi
if [ ! -d "$CALIBRATED_FRAMES_DIR" ] || [ -z "$(ls -A "$CALIBRATED_FRAMES_DIR" 2>/dev/null)" ]; then
    echo "Error: Calibrated frames directory '$CALIBRATED_FRAMES_DIR' is empty or not found. This is needed for image_path to split_for_mini.py. Please ensure it's populated in your mounted outputs/calibrated_frames."
    exit 1
fi
echo "All required inputs for mini-mosaic generation are present."


# --- PHASE 1: Split to group based on boundaries (using pre-generated data) ---
echo "Running split_for_mini.py to partition images..."
# Corrected path: assuming split_for_mini.py is directly in /app/code
split_cmd="python split_for_mini.py \
    -image_path \"$CALIBRATED_FRAMES_DIR\" \
    -save_path \"$MINI_PARTITION_DIR\" \
    -hm \"$HOMOGRAPHY_MATRICES_FILE\" \
    -angle_csv \"$ANGLE_CSV_FILE\""

if [ "$MODE_DUPLICATE" = "true" ]; then
    echo "Running split_for_mini.py script with the -duplicate flag."
    split_cmd="$split_cmd -duplicate"
elif [ "$MODE_DUPLICATE" = "false" ]; then
    echo "Running split_for_mini.py script without the -duplicate flag."
else
    echo "Warning: Unrecognized value for MODE_DUPLICATE. Expected 'true' or 'false'. Running without -duplicate."
fi

eval "$split_cmd"
if [ $? -ne 0 ]; then echo "Error: split_for_mini.py failed."; exit 1; fi
echo "Image partitioning complete."


# --- PHASE 2: Loop to mosaic all mini-mosaics ---
echo "Running stitcher.py for mini mosaics..."

# The H_asift_group_*.csv files are no longer directly used by stitcher.py,
# but 'find' is still used to iterate through the group folders.
find "$MINI_PARTITION_DIR" -name "H_asift_group_*.csv" | sort | while read mini_hm; do
    if [ -f "$mini_hm" ]; then # Ensure the file actually exists
        echo "Processing group homography file: $mini_hm" # This print is now informative, not critical for stitcher.py

        group_folder=$(basename "$(dirname "$mini_hm")")
        group_number=$(echo "$group_folder" | grep -oE '[0-9]+')

        # Corrected path: assuming stitcher.py is directly in /app/code
        # <<<<< START OF CHANGE: Removed -hm argument from stitcher_cmd >>>>>
        stitcher_cmd="python stitcher.py \
            -image_path \"$MINI_PARTITION_DIR/$group_folder\" \
            -save_path \"$MINI_MOSAICS_DIR\" \
            -scale 5 \
            -fname \"group$group_number\" \
            -mini_mosaic"
        # <<<<< END OF CHANGE >>>>>

        eval "$stitcher_cmd"
        if [ $? -ne 0 ]; then echo "Error: stitcher.py failed for $mini_hm."; break; fi
    else
        echo "Error: File not found after 'find' for $mini_hm" >&2
    fi
done

# The check for H_asift_group_*.csv files is still relevant to ensure
# that groups were created by split_for_mini.py, even if stitcher.py doesn't use them.
if ! find "$MINI_PARTITION_DIR" -name "H_asift_group_*.csv" -print -quit | grep -q .; then
    echo "WARNING: No H_asift_group_*.csv files were found in "$MINI_PARTITION_DIR". Mini mosaics will not be generated." >&2
    exit 1
fi
echo "Mini mosaics complete."

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Total Elapsed time for mini-mosaic pipeline: $elapsed seconds"
echo "Mini-mosaic generation finished successfully! Check outputs in $MINI_MOSAICS_DIR"
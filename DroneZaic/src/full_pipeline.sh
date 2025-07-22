#!/bin/bash
# This script automates the entire image processing pipeline:
# 1. Dynamic sampling (frame extraction)
# 2. Calibration
# 3. ASIFT Homography Estimation
# 4. Mini-mosaic generation (splitting and stitching)

set -e # Exit immediately if a command exits with a non-zero status.

# --- Define Base Paths and Parameters ---
APP_DIR="/app"
CONTAINER_OUTPUTS_DIR="/app/outputs" # This will be mounted from host
INPUT_VIDEOS_DIR="$APP_DIR/input_videos" # Directory where uploaded videos are copied
SRC_DIR="$APP_DIR/src" # New variable for the source code directory

# Check if video filename is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <video_filename> [srt_filename]"
    echo "Example: $0 DJI_0604.MOV DJI_0604.srt"
    exit 1
fi

VIDEO_FILENAME="$1"
SRT_FILENAME="$2" # Optional SRT filename

# Extract filename without extension for dynamic naming
VIDEO_BASENAME="${VIDEO_FILENAME%.*}"

# Construct full paths for input video and srt
VIDEO_INPUT_PATH="$INPUT_VIDEOS_DIR/$VIDEO_FILENAME"
SRT_INPUT_PATH=""
if [ -n "$SRT_FILENAME" ]; then
    SRT_INPUT_PATH="$INPUT_VIDEOS_DIR/$SRT_FILENAME"
fi

# Ensure the base outputs directory exists
mkdir -p "$CONTAINER_OUTPUTS_DIR"
echo "Ensured base output directory $CONTAINER_OUTPUTS_DIR is ready."

# --- PHASE 1: Dynamic Sampling (Frame Extraction) ---
echo "--- Starting Dynamic Sampling (Frame Extraction) ---"
DYNAMIC_SAMPLING_FRAMES_DIR="$CONTAINER_OUTPUTS_DIR/extracted_frames"
DYNAMIC_SAMPLING_CMD="python \"$SRC_DIR/dynamic_sampling.py\" \
    -video \"$VIDEO_INPUT_PATH\" \
    -save_path \"$DYNAMIC_SAMPLING_FRAMES_DIR\" \
    -win 100 \
    -scale 3 \
    -fname \"${VIDEO_BASENAME}_frames\" \
    -format png \
    -clear_output"

if [ -n "$SRT_INPUT_PATH" ]; then
    DYNAMIC_SAMPLING_CMD="$DYNAMIC_SAMPLING_CMD -srt \"$SRT_INPUT_PATH\""
fi

eval "$DYNAMIC_SAMPLING_CMD"
if [ $? -ne 0 ]; then echo "Error: dynamic_sampling.py failed."; exit 1; fi
echo "Dynamic sampling complete. Frames saved to $DYNAMIC_SAMPLING_FRAMES_DIR."

# --- PHASE 2: Calibration ---
echo "--- Starting Calibration ---"
CALIBRATED_FRAMES_DIR="$CONTAINER_OUTPUTS_DIR/calibrated_frames"
# Ensure the raw frames directory exists for calibration input
RAW_FRAMES_INPUT_PATH="$DYNAMIC_SAMPLING_FRAMES_DIR/raw"
if [ ! -d "$RAW_FRAMES_INPUT_PATH" ]; then
    echo "Error: Raw frames directory '$RAW_FRAMES_INPUT_PATH' not found. Dynamic sampling might have failed or saved frames elsewhere."
    exit 1
fi
python "$SRC_DIR/calibration.py" \
    -image_path "$RAW_FRAMES_INPUT_PATH" \
    -save_path "$CALIBRATED_FRAMES_DIR"
if [ $? -ne 0 ]; then echo "Error: calibration.py failed."; exit 1; fi
echo "Calibration complete. Calibrated frames saved to $CALIBRATED_FRAMES_DIR."

# --- PHASE 3: ASIFT Homography Estimation ---
echo "--- Starting ASIFT Homography Estimation ---"
HOMOGRAPHY_RESULTS_DIR="$CONTAINER_OUTPUTS_DIR/homography_results"
HM_METHOD="asift" # Explicitly using 'asift' for homography method
HOMOGRAPHY_MATRICES_FILE="$HOMOGRAPHY_RESULTS_DIR/homography_matrices/H_${HM_METHOD}.csv"

# --- IMPORTANT FIX HERE ---
# Add /app/src to PYTHONPATH so Python can find 'asift' package
# Then, use the module name 'asift.asift_homography_estimation' with -m
PYTHONPATH="$SRC_DIR:$PYTHONPATH" python -m asift.asift_homography_estimation \
    -image_path "$CALIBRATED_FRAMES_DIR" \
    -save_path "$HOMOGRAPHY_RESULTS_DIR" \
    -hm "$HOMOGRAPHY_MATRICES_FILE" \
    -scale 3
if [ $? -ne 0 ]; then echo "Error: asift_homography_estimation failed."; exit 1; fi
echo "ASIFT Homography Estimation complete. Homography matrices saved to $HOMOGRAPHY_MATRICES_FILE."

# --- PHASE 4: Mini-Mosaic Generation (from run_mini.sh content) ---
echo "--- Starting Mini-Mosaic Generation ---"

MODE_DUPLICATE="true" # Hardcoding mode_duplicate to 'true' as per your typical use

# Define paths for internal use within the container for mini-mosaics
# Angle CSV file name now dynamically uses the video basename
ANGLE_CSV_FILE="$DYNAMIC_SAMPLING_FRAMES_DIR/${VIDEO_BASENAME}_frames_angle_diff.csv"
MINI_PARTITION_DIR="$CONTAINER_OUTPUTS_DIR/${HM_METHOD}_mini_partition"
MINI_MOSAICS_DIR="$CONTAINER_OUTPUTS_DIR/${HM_METHOD}_mini_mosaics"

# --- Clear previous mini-mosaic and partition outputs for a clean run ---
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
echo "Ensured mini-mosaic output directories are ready."

start_mini_time=$(date +%s)

# --- VALIDATE REQUIRED INPUTS FOR MINI-MOSAIC GENERATION ---
if [ ! -f "$ANGLE_CSV_FILE" ]; then
    echo "Error: Angle CSV file '$ANGLE_CSV_FILE' not found. Please ensure it's generated by dynamic_sampling."
    exit 1
fi
if [ ! -f "$HOMOGRAPHY_MATRICES_FILE" ]; then
    echo "Error: Homography matrix file '$HOMOGRAPHY_MATRICES_FILE' not found. Please ensure it's generated by asift_homography_estimation."
    exit 1
fi
if [ ! -d "$CALIBRATED_FRAMES_DIR" ] || [ -z "$(ls -A "$CALIBRATED_FRAMES_DIR" 2>/dev/null)" ]; then
    echo "Error: Calibrated frames directory '$CALIBRATED_FRAMES_DIR' is empty or not found. This is needed for image_path to split_for_mini.py."
    exit 1
fi
echo "All required inputs for mini-mosaic generation are present."

# --- PHASE 4.1: Split to group based on boundaries ---
echo "Running split_for_mini.py to partition images..."
split_cmd="python \"$SRC_DIR/split_for_mini.py\" \
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

# --- PHASE 4.2: Loop to mosaic all mini-mosaics ---
echo "Running stitcher.py for mini mosaics..."

GROUP_HM_PREFIX="H_${HM_METHOD}_group_" # This will be "H_asift_group_"

find "$MINI_PARTITION_DIR" -name "${GROUP_HM_PREFIX}*.csv" | sort | while read mini_hm; do
    if [ -f "$mini_hm" ]; then
        echo "Processing group homography file: $mini_hm"

        group_folder=$(basename "$(dirname "$mini_hm")")
        group_number=$(echo "$group_folder" | grep -oE '[0-9]+')

        stitcher_cmd="python \"$SRC_DIR/stitcher.py\" \
            -image_path \"$MINI_PARTITION_DIR/$group_folder\" \
            -hm \"$mini_hm\" \
            -save_path \"$MINI_MOSAICS_DIR\" \
            -scale 4 \
            -fname \"group$group_number\" \
            -mini_mosaic"

        eval "$stitcher_cmd"
        if [ $? -ne 0 ]; then echo "Error: stitcher.py failed for $mini_hm."; break; fi
    else
        echo "Error: File not found after 'find' for $mini_hm" >&2
    fi
done

if ! find "$MINI_PARTITION_DIR" -name "${GROUP_HM_PREFIX}*.csv" -print -quit | grep -q .; then
    echo "WARNING: No ${GROUP_HM_PREFIX}*.csv files were found in "$MINI_PARTITION_DIR". Mini mosaics will not be generated." >&2
    exit 1
fi
echo "Mini mosaics complete."

end_mini_time=$(date +%s)
elapsed_mini=$(( end_mini_time - start_mini_time ))

echo "Total Elapsed time for mini-mosaic pipeline: $elapsed_mini seconds"
echo "Full pipeline finished successfully! Check outputs in $CONTAINER_OUTPUTS_DIR."

#!/bin/bash

# Define variables for paths and methods
CONTAINER_OUTPUTS_DIR="/app/outputs" # This will be mounted from host

hm_method=""
mode_duplicate=true
TILE_SIZE_PX=4096 # Define default tile size here
OUTPUT_GLOBAL_MOSAIC_NAME="final_global_mosaic" # Define default output name
ASIFT_SCALE=1.0 # ADD THIS LINE: Define default ASIFT scale here

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        # -p|--working_path is now ignored, as outputs are handled by Docker volume
        -p|--working_path)
            echo "Warning: -p|--working_path is ignored. Outputs are now managed by Docker volume at $CONTAINER_OUTPUTS_DIR."
            shift ;;
        -h|--hm_method) hm_method="$2"; shift ;;
        -d|--mode_duplicate) mode_duplicate="$2"; shift ;; # Use true or false
        -t|--tile_size) TILE_SIZE_PX="$2"; shift ;; # New argument for tile size
        -o|--output_name) OUTPUT_GLOBAL_MOSAIC_NAME="$2"; shift ;; # New argument for output name
        # ADD THIS BLOCK: Handle the -a|--asift_scale argument
        -a|--asift_scale) ASIFT_SCALE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$hm_method" ]; then
    echo "Usage: ./maizaic_run.sh -h <homography_method> [-d <mode_duplicate>] [-t <tile_size_pixels>] [-o <output_name>] [-a <asift_scale>]"
    echo "Example: ./maizaic_run.sh -h surf -d false -t 8192 -o my_big_mosaic -a 0.5"
    exit 1
fi

# Define paths for internal use within the container, ASSUMING PREVIOUS STAGES HAVE RUN
DYNAMIC_SAMPLING_FRAMES_DIR="$CONTAINER_OUTPUTS_DIR/extracted_frames" # Expected to exist
ANGLE_CSV_FILE="$DYNAMIC_SAMPLING_FRAMES_DIR/DJI_0604_frames_angle_diff.csv" # Expected to exist
CALIBRATED_FRAMES_DIR="$CONTAINER_OUTPUTS_DIR/calibrated_frames" # Expected to exist (input for split_for_mini)
HOMOGRAPHY_RESULTS_DIR="$CONTAINER_OUTPUTS_DIR/homography_results" # Expected to exist
HOMOGRAPHY_MATRICES_FILE="$HOMOGRAPHY_RESULTS_DIR/homography_matrices/H_${hm_method}.csv" # Expected to exist


# Ensure output subdirectories for THIS script's new outputs exist
mkdir -p "$CONTAINER_OUTPUTS_DIR/${hm_method}_mini_partition"
mkdir -p "$CONTAINER_OUTPUTS_DIR/${hm_method}_mini_mosaics"
mkdir -p "$CONTAINER_OUTPUTS_DIR/${hm_method}_global_mosaic_tiled" # Changed name to reflect tiling

start_time=$(date +%s)

# --- VALIDATE REQUIRED INPUTS FOR THIS SCRIPT'S STARTING POINT ---
# We assume calibrated_frames_dir, angle_csv, and homography matrix file are present.
# Checks for calibrated_frames_dir content removed as requested.
if [ ! -f "$ANGLE_CSV_FILE" ]; then
    echo "Error: Angle CSV file '$ANGLE_CSV_FILE' not found. Please ensure it's generated in your mounted outputs/extracted_frames."
    exit 1
fi
if [ ! -f "$HOMOGRAPHY_MATRICES_FILE" ]; then
    echo "Error: Homography matrix file '$HOMOGRAPHY_MATRICES_FILE' not found. Please ensure it's generated in your mounted outputs/homography_results."
    exit 1
fi
if [ ! -d "$CALIBRATED_FRAMES_DIR" ] || [ -z "$(ls -A "$CALIBRATED_FRAMES_DIR" 2>/dev/null)" ]; then
    echo "Error: Calibrated frames directory '$CALIBRATED_FRAMES_DIR' is empty or not found. This is needed for image_path to split_for_mini.py. Please ensure it's populated in your mounted outputs/calibrated_frames."
    exit 1
fi

# --- PHASE 1: Split to group based on boundaries (using pre-generated data) ---
echo "Running split_for_mini.py to partition images..."
split_cmd="python code/split_for_mini.py \
    -image_path \"$CALIBRATED_FRAMES_DIR\" \
    -save_path \"$CONTAINER_OUTPUTS_DIR/${hm_method}_mini_partition\" \
    -hm \"$HOMOGRAPHY_MATRICES_FILE\" \
    -angle_csv \"$ANGLE_CSV_FILE\""

if [ "$mode_duplicate" = "true" ]; then
    echo "Running split_for_mini.py script with the -duplicate flag."
    split_cmd="$split_cmd -duplicate"
elif [ "$mode_duplicate" = "false" ]; then
    echo "Running split_for_mini.py script without the -duplicate flag."
else
    echo "Warning: Unrecognized value for mode_duplicate. Expected 'true' or 'false'. Running without -duplicate."
fi

eval "$split_cmd"
if [ $? -ne 0 ]; then echo "Error: split_for_mini.py failed."; exit 1; fi
echo "Image partitioning complete."


# --- PHASE 2: Loop to mosaic all mini-mosaics ---
echo "Running stitcher.py for mini mosaics..."

MINI_PARTITION_DIR="$CONTAINER_OUTPUTS_DIR/${hm_method}_mini_partition"
MINI_MOSAICS_DIR="$CONTAINER_OUTPUTS_DIR/${hm_method}_mini_mosaics"

find "$MINI_PARTITION_DIR" -name "H_asift_group_*.csv" | sort | while read mini_hm; do
    if [ -f "$mini_hm" ]; then
        echo "Processing $mini_hm"

        group_folder=$(basename "$(dirname "$mini_hm")")
        group_number=$(echo "$group_folder" | grep -oE '[0-9]+')

        stitcher_cmd="python code/stitcher.py \
            -image_path \"$MINI_PARTITION_DIR/$group_folder\" \
            -hm \"$mini_hm\" \
            -save_path \"$MINI_MOSAICS_DIR\" \
            -scale 3 \
            -fname \"group$group_number\" \
            -mini_mosaic"

        eval "$stitcher_cmd"
        if [ $? -ne 0 ]; then echo "Error: stitcher.py failed for $mini_hm."; break; fi
    else
        echo "Error: File not found after 'find' for $mini_hm" >&2
    fi
done

if ! find "$MINI_PARTITION_DIR" -name "H_asift_group_*.csv" -print -quit | grep -q .; then
    echo "WARNING: No H_asift_group_*.csv files were found in $MINI_PARTITION_DIR. Mini mosaics will not be generated." >&2
    exit 1
fi
echo "Mini mosaics complete."

# --- PHASE 3: Assemble global mosaic using out-of-core tiling ---
echo "Assembling global mosaic using out-of-core tiling..."

GLOBAL_MOSAIC_TILED_DIR="$CONTAINER_OUTPUTS_DIR/${hm_method}_global_mosaic_tiled" # Use the new directory name

if ! ls "$MINI_MOSAICS_DIR"/*.png &> /dev/null; then
    echo "Error: No mini mosaics found in '$MINI_MOSAICS_DIR'. Cannot assemble global mosaic."
    exit 1
fi

python code/mini_mosaic_360.py \
    -image_path "$MINI_MOSAICS_DIR" \
    -save_path "$GLOBAL_MOSAIC_TILED_DIR" \
    -tile_size_px "$TILE_SIZE_PX" \
    -output_name "$OUTPUT_GLOBAL_MOSAIC_NAME" \
    -asift_scale "$ASIFT_SCALE" # ADD THIS LINE: Pass the ASIFT scale to mini_mosaic_360.py

if [ $? -ne 0 ]; then echo "Error: mini_mosaic_360.py failed to assemble global mosaic."; exit 1; fi
echo "Global mosaic assembly complete."

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Total Elapsed time for pipeline: $elapsed seconds"
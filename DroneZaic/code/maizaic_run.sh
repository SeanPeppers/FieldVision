#!/bin/bash

# Define variables for paths and methods
CONTAINER_OUTPUTS_DIR="/app/outputs" # This will be mounted from host

hm_method=""
mode_duplicate=true

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        # -p|--working_path is now ignored, as outputs are handled by Docker volume
        -p|--working_path)
            echo "Warning: -p|--working_path is ignored. Outputs are now managed by Docker volume at $CONTAINER_OUTPUTS_DIR."
            shift ;;
        -h|--hm_method) hm_method="$2"; shift ;;
        -d|--mode_duplicate) mode_duplicate="$2"; shift ;; # Use true or false
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$hm_method" ]; then
    echo "Usage: ./maizaic_run.sh -h <homography_method> [-d <mode_duplicate>]"
    echo "Example: ./maizaic_run.sh -h surf -d false"
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
mkdir -p "$CONTAINER_OUTPUTS_DIR/${hm_method}_global_mosaic"

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

MINI_PARTITION_DIR="$CONTAINER_OUTPUTS_DIR/${hm_method}_mini_partition" # Ensure this path is defined
MINI_MOSAICS_DIR="$CONTAINER_OUTPUTS_DIR/${hm_method}_mini_mosaics" # Ensure this path is defined

# Corrected find command to look for H_asift_group_*.csv
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
        if [ $? -ne 0 ]; then echo "Error: stitcher.py failed for $mini_hm."; break; fi # Break if stitcher fails
    else
        echo "Error: File not found after 'find' for $mini_hm" >&2
    fi
done

# Handle the case if find returns no files
# Corrected find command here as well
if ! find "$MINI_PARTITION_DIR" -name "H_asift_group_*.csv" -print -quit | grep -q .; then
    echo "WARNING: No H_asift_group_*.csv files were found in $MINI_PARTITION_DIR. Mini mosaics will not be generated." >&2
    exit 1 # Exit if mini_mosaics are critical and not found
fi
echo "Mini mosaics complete."

# --- PHASE 3: Assemble global mosaic ---
echo "Assembling global mosaic..."

GLOBAL_MOSAIC_DIR="$CONTAINER_OUTPUTS_DIR/${hm_method}_global_mosaic" # Ensure this path is defined

# Ensure the MINI_MOSAICS_DIR has content before proceeding to global mosaic
if ! ls "$MINI_MOSAICS_DIR"/*.png &> /dev/null; then # Check for at least one PNG file
    echo "Error: No mini mosaics found in '$MINI_MOSAICS_DIR'. Cannot assemble global mosaic."
    exit 1
fi

python code/mini_mosaic_360.py \
    -image_path "$MINI_MOSAICS_DIR" \
    -save_path "$GLOBAL_MOSAIC_DIR"

if [ $? -ne 0 ]; then echo "Error: mini_mosaic_360.py failed to assemble global mosaic."; exit 1; fi
echo "Global mosaic assembly complete."

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Total Elapsed time for pipeline: $elapsed seconds"
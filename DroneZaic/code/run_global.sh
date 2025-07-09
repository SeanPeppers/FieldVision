#!/bin/bash
# This script is designed to run ONLY the global mosaic assembly phase (mini_mosaic_360.py)
# It assumes all prior steps (calibration, homography estimation, image partitioning, and mini-mosaic generation)
# have already been successfully completed and their outputs are present.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Define Paths and Parameters ---
APP_DIR="/app"
CONTAINER_OUTPUTS_DIR="/app/outputs" # This will be mounted from host

# Specific directories and files for the global mosaic phase
HM_METHOD="surf" # Homography method used for mini-mosaics, needs to match your setup
MINI_MOSAICS_INPUT_DIR="$CONTAINER_OUTPUTS_DIR/${HM_METHOD}_mini_mosaics" # Input for global mosaic
GLOBAL_MOSAIC_OUTPUT_DIR="$CONTAINER_OUTPUTS_DIR/${HM_METHOD}_global_mosaic_tiled" # Output of global mosaic

# Parameters for mini_mosaic_360.py (can be adjusted here)
TILE_SIZE_PX=4096 # Size of square tiles in pixels
OUTPUT_GLOBAL_MOSAIC_NAME="final_global_mosaic" # Base name for the final GeoTIFF
ASIFT_SCALE=0.25 # Scale factor for images passed to ASIFT in mini_mosaic_360.py
                 # Adjust this value (e.g., 0.1, 0.5, 0.75, 1.0) for tuning global stitch quality.

echo "--- Starting ONLY Global Mosaic Assembly ---"

# --- Clear previous global mosaic output for a clean run ---
echo "--- Clearing previous global mosaic tiled output: $GLOBAL_MOSAIC_OUTPUT_DIR ---"
if [ -d "$GLOBAL_MOSAIC_OUTPUT_DIR" ]; then
    rm -rf "$GLOBAL_MOSAIC_OUTPUT_DIR"
    echo "Removed existing $GLOBAL_MOSAIC_OUTPUT_DIR."
fi
# Ensure the output directory exists for the new run
mkdir -p "$GLOBAL_MOSAIC_OUTPUT_DIR"
echo "Ensured $GLOBAL_MOSAIC_OUTPUT_DIR is ready."

# --- Validate Required Inputs ---
# This check ensures that the mini-mosaics (which are inputs for this script) exist before proceeding.
if ! ls "$MINI_MOSAICS_INPUT_DIR"/*.png &> /dev/null; then
    echo "Error: No mini mosaics found in '$MINI_MOSAICS_INPUT_DIR'."
    echo "This script requires pre-generated mini-mosaics from previous pipeline stages."
    echo "Please run the full pipeline (e.g., 'run_maizaic_pipeline.sh') first, or ensure '$MINI_MOSAICS_INPUT_DIR' is populated."
    exit 1
fi
echo "Input mini-mosaics found in $MINI_MOSAICS_INPUT_DIR."

# --- Run Global Mosaic Assembly ---
echo "--- Assembling global mosaic using out-of-core tiling (via mini_mosaic_360.py) ---"
cd "$APP_DIR" # Ensure we are in the application root directory

# Call maizaic_run.sh with all its expected arguments
# Note: The -p argument in maizaic_run.sh is just a warning, it's not strictly necessary.
# The core arguments are -h, -d, -t, -o, and -a (if implemented in maizaic_run.sh)
./code/maizaic_run.sh \
    -h "$HM_METHOD" \
    -d false \
    -t "$TILE_SIZE_PX" \
    -o "$OUTPUT_GLOBAL_MOSAIC_NAME" \
    -a "$ASIFT_SCALE" \
    -p "$CONTAINER_OUTPUTS_DIR" # Moved -p to the end, as it's often ignored anyway

if [ $? -ne 0 ]; then
    echo "Error: Global mosaic assembly failed. Check logs above for details from mini_mosaic_360.py."
    exit 1
fi

echo "--- Global mosaic assembly complete! ---"
echo "Final global mosaic should be in: $GLOBAL_MOSAIC_OUTPUT_DIR/$OUTPUT_GLOBAL_MOSAIC_NAME.tif"
echo "Script finished successfully!"
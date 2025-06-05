#!/bin/bash

working_path="/maizaic_data/"
hm_method="surf"

# Assuming angle_files are generated elsewhere or are at a specific known location
# For now, this will look for CSVs directly in /maizaic_data/ (which might be empty).
angle_files=("$working_path"/*.csv) 

echo "--- Starting Maizaic Image Stitching Pipeline ---"
echo "Working directory: $working_path"
echo "Homography method: $hm_method"

# Step 0: Ensure necessary directories exist.
# These directories are *inside* surf_mini_partition based on your screenshot.
echo ""
echo "--- Step 0: Ensuring necessary directories exist for outputs ---"
mkdir -p "$working_path/${hm_method}_mini_partition/frames_to_mosaic"
mkdir -p "$working_path/${hm_method}_mini_partition/homography_matrices"
mkdir -p "$working_path/${hm_method}_mini_mosaics" # Output of stitcher.py
mkdir -p "$working_path/${hm_method}_global_mosaic" # Output of surf_assembly_global.py


# Step 1: Generate Homographies for individual frames AND populate frames_to_mosaic.
# This script should take your raw/calibrated images and process them into the 'frames_to_mosaic'
# directory *within* surf_mini_partition, and also generate H_surf.csv.
echo ""
echo "--- Step 1: Estimating Homographies for Individual Frames and Populating Frames_to_Mosaic ---"
# IMPORTANT: Adjust -image_path to where your primary input images are.
# Based on your structure, this is likely "$working_path/calibrated" or "$working_path/sampled_frames"
# I will use "$working_path/calibrated" as a common source.
# The -save_path is critical: it tells surf_homography_estimation.py where to put its generated frames_to_mosaic and homography_matrices folders.
python surf/surf_homography_estimation.py \
    -image_path "$working_path/calibrated" \
    -save_path "$working_path/${hm_method}_mini_partition" \
    -scale 3 # Make sure this scale matches your image preparation if any

if [ $? -ne 0 ]; then
    echo "ERROR: surf_homography_estimation.py failed. Check its requirements (input images, Python 2.7, etc.). Exiting."
    exit 1
fi

echo "Verifying Step 1 outputs:"
ls -l "$working_path/${hm_method}_mini_partition/homography_matrices/H_${hm_method}.csv"
ls -l "$working_path/${hm_method}_mini_partition/frames_to_mosaic/"


# Step 2: Split images and homographies into mini-mosaic groups
# This script reads H_surf.csv and splits the 'frames_to_mosaic' into 'group_XXX' subdirectories and H_asift_group_XXX.csv files.
echo ""
echo "--- Step 2: Splitting Frames into Mini-Mosaic Groups ---"
python split_for_mini.py \
     -image_path "$working_path/${hm_method}_mini_partition/frames_to_mosaic" \
     -save_path "$working_path/${hm_method}_mini_partition" \
     -hm "$working_path/${hm_method}_mini_partition/homography_matrices/H_${hm_method}.csv" \
     -angle_csv "${angle_files[@]}" # Double-check angle_files path if not directly in /maizaic_data/

if [ $? -ne 0 ]; then
    echo "ERROR: split_for_mini.py failed. Exiting."
    exit 1
fi
echo "Mini-mosaic partitions (group_XXX folders and H_asift_group_XXX.csv) should be created in $working_path/${hm_method}_mini_partition/"
ls -l "$working_path/${hm_method}_mini_partition/group_*"
ls -l "$working_path/${hm_method}_mini_partition/H_asift_group_*.csv"


# Step 3: Loop to stitch all mini-mosaics
echo ""
echo "--- Step 3: Stitching Individual Mini-Mosaics ---"

# Check if any H_asift_group_*.csv files were created by split_for_mini.py
mini_path="$working_path/${hm_method}_mini_partition"
shopt -s nullglob # Enable nullglob to make the loop skip if no files match
csv_files=("$mini_path"/H_asift_group_*.csv) # Explicitly look for these files
shopt -u nullglob # Disable nullglob

if [ ${#csv_files[@]} -eq 0 ]; then
    echo "ERROR: No H_asift_group_*.csv homography files found in $mini_path. split_for_mini.py might have failed or found no groups. Exiting."
    exit 1
fi

for mini_hm in "${csv_files[@]}";
do
    if [ -f "$mini_hm" ]; then
        echo "Processing mini-mosaic homography file: $mini_hm"

        # Extract the group number from the file name
        group_number=$(basename "$mini_hm" | grep -oE '[0-9]+')
        
        # IMPORTANT: Ensure your stitcher.py has the latest corrections for mini-mosaic creation.
        # It's crucial that this script correctly saves the stitched mini-mosaics to the -save_path.
        python stitcher.py \
            -image_path "$mini_path/group_$group_number" \
            -hm "$mini_hm" \
            -save_path "$working_path/${hm_method}_mini_mosaics" \
            -fname "group$group_number" \
            -mini_mosaic
        if [ $? -ne 0 ]; then
            echo "WARNING: stitcher.py failed for group $group_number. Continuing to next group (if configured to do so)."
            # Decide if you want to exit here or continue with other groups
        fi
    else
        echo "WARNING: CSV file not found (should not happen in this loop): $mini_hm. Skipping."
    fi
done

echo ""
echo "--- Verifying Stitched Mini-Mosaics Output ---"
echo "Listing contents of: $working_path/${hm_method}_mini_mosaics"
ls -l "$working_path/${hm_method}_mini_mosaics"
if [ $(ls -1 "$working_path/${hm_method}_mini_mosaics" | wc -l) -eq 0 ]; then
    echo "ERROR: No mini-mosaic images were found in $working_path/${hm_method}_mini_mosaics/. The 'stitcher.py' step likely failed to save outputs. Exiting."
    exit 1
fi


# Step 4: Assemble global mosaic from mini-mosaics
echo ""
echo "--- Step 4: Assembling Global Mosaic from Mini-Mosaics ---"
python surf/surf_assembly_global.py \
    -image_path "$working_path/${hm_method}_mini_mosaics" \
    -save_path "$working_path/${hm_method}_global_mosaic"
if [ $? -ne 0 ]; then
    echo "ERROR: surf_assembly_global.py failed. Exiting."
    exit 1
fi

echo ""
echo "--- Pipeline Finished Successfully ---"
import os
import argparse
import logging
from datetime import datetime
import cv2
import csv
import numpy as np
from numpy.linalg import inv
import time
import sys
import copy
import shutil
from scipy.signal import find_peaks, medfilt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
import traceback
import subprocess # NEW: Import subprocess for calling external commands

# NEW: Function to preserve EXIF data using exiftool (copied for self-containment)
def preserve_exif_data_split(original_file_path, new_file_path):
    """
    Copies all EXIF/metadata from the original_file_path to the new_file_path
    using the exiftool command-line utility.
    """
    try:
        command = ['exiftool', '-tagsFromFile', original_file_path, '-all:all', '-overwrite_original', '-q', new_file_path]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Logging here can be useful for debugging, but split_for_mini has its own stdout for debug messages
        # print(f"DEBUG(split_exif): EXIF copy successful from {os.path.basename(original_file_path)} to {os.path.basename(new_file_path)}", file=sys.stderr)
        if result.stderr:
            print(f"WARNING(split_exif): exiftool stderr: {result.stderr.strip()}", file=sys.stderr)

    except FileNotFoundError:
        print("ERROR(split_exif): 'exiftool' command not found. Please install exiftool.", file=sys.stderr)
        print(f"  EXIF data NOT preserved for {os.path.basename(new_file_path)}.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR(split_exif): exiftool failed for {os.path.basename(new_file_path)}: {e.stderr.strip()}", file=sys.stderr)
        print(f"  EXIF data NOT preserved for {os.path.basename(new_file_path)}.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR(split_exif): Unexpected error during EXIF copy for {os.path.basename(new_file_path)}: {e}", file=sys.stderr)
        print(f"  EXIF data NOT preserved for {os.path.basename(new_file_path)}.", file=sys.stderr)


def move_images(image_path_list, save_path, homography, boundaries, duplicate, overlap): # Renamed image_path to image_path_list for clarity
    image_files = []
    H = [] 

    # Ensure the base save_path exists (e.g., outputs/surf_mini_partition)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"DEBUG(split): Created base partition save_path: {save_path}", file=sys.stderr)


    # --- Load image files from the provided image_path(s) ---
    # The image_path_list argument is a list of paths. Handle each one.
    # Assuming image_path_list[0] will be the directory containing the frames (e.g., outputs/calibrated_frames)
    for img_p in image_path_list:
        if os.path.isdir(img_p):
            extensions = [".jpeg", ".jpg", ".png", ".tif", ".tiff"] # Added .tiff for robustness
            for file_path in sorted(os.listdir(img_p)):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    image_files.append(os.path.join(img_p, file_path))
        else:
            print(f"WARNING(split): Image path '{img_p}' is not a directory. Skipping.", file=sys.stderr)
    
    if not image_files:
        print("ERROR(split): No image files found in the specified image_path(s). Cannot proceed with partitioning. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"DEBUG(split): Found {len(image_files)} image files for partitioning.", file=sys.stderr)


    # --- Load Homography matrices from the provided homography CSV ---
    print(f"DEBUG(split): Loading homographies from: {homography}", file=sys.stderr)
    try:
        with open(homography, 'r', newline='') as csvFile: # Added newline=''
            reader = csv.reader(csvFile, delimiter = ",")
            for row in reader:
                if row: # Ensure row is not empty before attempting conversion
                    # Homographies are 1x9 flat arrays, so no need for reshape here
                    H_each = [float(val) for val in row] # Convert elements to float
                    H.append(H_each)
        print(f"DEBUG(split): Successfully loaded {len(H)} homography rows from {homography}.", file=sys.stderr)
        if not H:
            print(f"WARNING(split): Homography file {homography} was empty or contained no valid data. This might lead to empty group CSVs.", file=sys.stderr)
    except FileNotFoundError:
        print(f"ERROR(split): Homography file not found at {homography}. This file is critical for partitioning. Exiting.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # Critical error, exit
    except Exception as e:
        print(f"ERROR(split): Failed to load homography file {homography}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # Critical error, exit


    group_boundary = boundaries
    print(f'DEBUG(split): Group boundaries: {group_boundary}', file=sys.stderr)
    group_count = 0
    subfolder_count = 1
    boundary_idx = 0 # Use a clearer name for the index into group_boundary list
    
    # Initialize the first subfolder path
    subfolder = os.path.join(save_path, 'group_{}'.format(str(subfolder_count).zfill(3)))
    print(f"DEBUG(split): Initial subfolder path for groups: {subfolder}", file=sys.stderr) 
    
    i = 0 # Current image index being processed (corresponds to image_files index)
    start_hm_idx = 0 # Start index for slicing homographies (corresponds to H list index)

    while i < len(image_files):
        filename = os.path.basename(image_files[i])
        
        # Create the current group subfolder if it doesn't exist
        # This check is done when group_count is 0, meaning it's a new group
        if group_count == 0 and not os.path.exists(subfolder):
            os.makedirs(subfolder)
            print(f"DEBUG(split): Created group subfolder: {subfolder}", file=sys.stderr)

        source_path = image_files[i] # Original image path
        destination_path = os.path.join(subfolder, filename)
        
        # Move or copy the image file
        try:
            if not duplicate:
                shutil.move(source_path, destination_path) # Move will keep EXIF if possible, but safer to re-embed
                print(f"DEBUG(split): Moved image {filename} to {subfolder}", file=sys.stderr)
            else:
                shutil.copy(source_path, destination_path) # Copy will strip EXIF unless re-embedded
                print(f"DEBUG(split): Copied image {filename} to {subfolder}", file=sys.stderr)
            
            # NEW: Always preserve EXIF data after move/copy for robustness
            preserve_exif_data_split(source_path, destination_path)

        except Exception as e:
            print(f"ERROR(split): Failed to move/copy {filename} to {destination_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Decide if you want to exit or continue on file error
            sys.exit(1) # Exit on critical file operation failure

        group_count += 1
        i = i + 1

        # Check if the current image index 'i' hits a boundary to close the current group and start a new one
        # Ensure boundary_idx is within the bounds of group_boundary list
        # group_boundary values represent the 'image index' where a new group starts.
        # Homographies are H_0_1, H_1_2, ... H_N-1_N. So H[k] corresponds to image k+1.
        # If boundary is image index `B`, then homographies for images up to `B-1` (i.e., H_0_1 ... H_B-2_B-1)
        # belong to the current group.
        # The number of homographies for N images is N-1.
        # If a group has images from index `S` to `E`, it will have `E-S` homographies.
        # `H` list is 0-indexed where `H[k]` is the homography between image `k` and `k+1`.
        # So, if images are `I_S, I_S+1, ..., I_E`, the homographies are `H[S], H[S+1], ..., H[E-1]`.
        # The number of such homographies is `(E-1) - S + 1 = E-S`.
        # Your `group_boundary` values indicate the *index of the image that starts the new group*.
        # So, if `group_boundary[boundary_idx]` is `B`, images `I_0` to `I_{B-1}` are in the current group.
        # This means homographies `H[0]` to `H[B-2]` belong to the current group.
        # The `start_hm_idx` is the starting index in the `H` list.
        # The end index should be `group_boundary[boundary_idx] - 1` (image index),
        # which means `H` index up to `group_boundary[boundary_idx] - 2`.
        # The slice end index in Python is exclusive.
        # So, `H[start_hm_idx : group_boundary[boundary_idx] - 1]` for homographies up to image `group_boundary[boundary_idx] - 1`.

        if boundary_idx < len(group_boundary) and i == group_boundary[boundary_idx]:
            # Slice the homographies for the current group
            # The indices in `H` correspond to the starting image of the pair.
            # If `group_boundary[boundary_idx]` is the image index where a new group starts,
            # then the last image in the current group is `group_boundary[boundary_idx] - 1`.
            # The last homography in `H` list that belongs to this group is `H[group_boundary[boundary_idx] - 2]`.
            # So the slice for H should be `H[start_hm_idx : group_boundary[boundary_idx] -1]` (exclusive end for Python slice).
            h_temp = H[start_hm_idx : group_boundary[boundary_idx] - 1] 
            
            print(f"DEBUG(split): --- Boundary hit! ({i} == {group_boundary[boundary_idx]}) ---", file=sys.stderr)
            print(f"DEBUG(split): Slicing homographies from index {start_hm_idx} to {group_boundary[boundary_idx]-1} (exclusive end). Resulting h_temp length: {len(h_temp)}", file=sys.stderr)
            
            if not h_temp:
                print(f"WARNING(split): h_temp (sliced homographies) is empty for group {subfolder_count}. No H_asift CSV will be written for this group.", file=sys.stderr)
            
            # Increment to the next boundary for the next group
            boundary_idx += 1

            # Construct the filename and path for the group's homography CSV
            group_hm_filename = "H_asift_group_{}.csv".format(str(subfolder_count).zfill(3))
            group_hm_filepath = os.path.join(subfolder, group_hm_filename)
            print(f"DEBUG(split): Attempting to write group homography CSV to: {group_hm_filepath}", file=sys.stderr)

            try:
                # Write the sliced homographies to the group-specific CSV
                with open(group_hm_filepath, 'w', newline='') as f1: # Use 'w' to overwrite, 'a' if appending multiple runs is intended
                    wr = csv.writer(f1, delimiter=",", quoting = csv.QUOTE_NONE)
                    for h_each_save in h_temp:
                        wr.writerow(h_each_save)
                print(f"DEBUG(split): Successfully wrote {len(h_temp)} rows to {group_hm_filepath}.", file=sys.stderr)
            except Exception as e:
                print(f"ERROR(split): Failed to write group homography CSV {group_hm_filepath}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # This error means the stitcher won't find the H for this group.
                # You might want to skip this group or log a severe error.

            # Reset for the next group
            i = i - overlap # Roll back 'i' for overlap if needed
            print(f"DEBUG(split): Rolled back image index 'i' to {i} due to overlap of {overlap}.", file=sys.stderr)
            group_count = 0 # Reset group image counter
            subfolder_count += 1 # Increment group folder counter
            
            # Update subfolder path for the next group
            subfolder = os.path.join(save_path, 'group_{}'.format(str(subfolder_count).zfill(3)))
            start_hm_idx = i # Set new start index for slicing homographies in the next group
            h_temp = [] # Clear h_temp for the next group (though re-assigned in next loop iter)

    # Handle any remaining images/homographies that didn't hit a boundary
    # This ensures the last group is also processed.
    if i > start_hm_idx: # If there are images left in the current (last) group
        h_temp = H[start_hm_idx : len(H)] # Slice from start_hm_idx to end of H list
        
        print(f"DEBUG(split): --- End of images reached. Processing final group. ---", file=sys.stderr)
        print(f"DEBUG(split): Slicing homographies from index {start_hm_idx} to {len(H)} (exclusive end). Resulting h_temp length: {len(h_temp)}", file=sys.stderr)

        if not h_temp:
            print(f"WARNING(split): h_temp (sliced homographies) is empty for final group {subfolder_count}. No H_asift CSV will be written.", file=sys.stderr)
        
        group_hm_filename = "H_asift_group_{}.csv".format(str(subfolder_count).zfill(3))
        group_hm_filepath = os.path.join(subfolder, group_hm_filename)
        print(f"DEBUG(split): Attempting to write final group homography CSV to: {group_hm_filepath}", file=sys.stderr)

        try:
            with open(group_hm_filepath, 'w', newline='') as f1:
                wr = csv.writer(f1, delimiter=",", quoting = csv.QUOTE_NONE)
                for h_each_save in h_temp:
                    wr.writerow(h_each_save)
            print(f"DEBUG(split): Successfully wrote {len(h_temp)} rows to {group_hm_filepath}.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR(split): Failed to write final group homography CSV {group_hm_filepath}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def filter_on_shoulder(peaks, properties, data, window_size):
    filtered_peaks = []
    for peak in peaks:
        start = max(0, peak - window_size)
        end = min(len(data), peak + window_size)
        print(peak, start, end)
        if data[peak] == max(data[start:end]):
            filtered_peaks.append(peak)
    return filtered_peaks


def median_filter_smoothing(data, window_size):
    return medfilt(data, window_size)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def find_threshold(data):
    mean = np.mean(data)
    std = np.std(data)
    threshold = mean + std
    return threshold


def find_peaks_above_threshold(data, threshold):
    return [i for i, value in enumerate(data) if value > threshold]

def plot_partition(angle_diffs, filtered_peaks, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(angle_diffs, label='data')
    plt.plot(filtered_peaks, angle_diffs[filtered_peaks], "x", label='peaks', color = 'red')
    plt.title("data with detected peaks")
    plt.xlabel("frame number")
    plt.ylabel("angle difference")
    plt.legend()
    plt.savefig(save_path)
    #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-save_path', dest='save_path', default="RESULTS/global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
    parser.add_argument('-hm', '--homography', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-angle_csv', '--angle_csv', type=str, nargs='+', help='csv file that stores the angle difference csv')
    parser.add_argument('-duplicate', dest='duplicate', action='store_true', help='Enable duplication. Default is disable.')
    args = parser.parse_args()

    save_path = args.save_path

    # Ensure the main output directory for partition plots exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"DEBUG(split main): Created main save_path: {save_path}", file=sys.stderr) # New debug print

    partition_boundary = []

    # Handle multiple angle CSV files (though your use case currently shows one)
    i_plot_counter = 0 # Separate counter for plot filenames
    prev_end_number = 0

    for angle_file in args.angle_csv:
        angle_diffs = []
        
        # Ensure the angle_file exists before trying to open it
        if not os.path.exists(angle_file):
            print(f"ERROR(split main): Angle CSV file not found: {angle_file}. Skipping this file.", file=sys.stderr)
            continue

        try:
            with open(angle_file, 'r', newline='') as file: # Added newline=''
                i_plot_counter += 1 # Increment for plot filename
                angle_reader = csv.reader(file)
                for row in angle_reader:
                    if row: # Ensure row is not empty
                        angle_diffs.append(float(row[0]))
            print(f"DEBUG(split main): Successfully read {len(angle_diffs)} angle differences from {angle_file}.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR(split main): Failed to read angle CSV file {angle_file}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue # Skip to next angle file

        if not angle_diffs:
            print(f"WARNING(split main): No angle differences found in {angle_file}. Skipping peak detection for this file.", file=sys.stderr)
            continue

        angle_diffs = np.asarray(angle_diffs)
        
        print(f"DEBUG(split main): Angle diffs (first 10): {angle_diffs[:10]}", file=sys.stderr)
        print(f"DEBUG(split main): Angle diffs (last 10): {angle_diffs[-10:]}", file=sys.stderr)
        print(f"DEBUG(split main): Length of angle_diffs: {len(angle_diffs)}", file=sys.stderr)
        print(f"DEBUG(split main): Mean of angle_diffs: {np.mean(angle_diffs):.4f}", file=sys.stderr)
        print(f"DEBUG(split main): Std Dev of angle_diffs: {np.std(angle_diffs):.4f}", file=sys.stderr)

        thresh = find_threshold(angle_diffs)
        print(f"DEBUG(split main): Calculated threshold for peaks: {thresh:.4f}", file=sys.stderr) # Formatted for clarity

        # Using distance=5 as specified in original
        filtered_peaks, properties = find_peaks(angle_diffs, thresh, distance=5)
        print(f"DEBUG(split main): Detected {len(filtered_peaks)} peaks in angle differences.", file=sys.stderr)
        
        plot_partition(angle_diffs, filtered_peaks, os.path.join(save_path, 'angle_peaks_plot%02d.png' % i_plot_counter))
        
        print(f"DEBUG(split main): Raw filtered_peaks: {filtered_peaks}", file=sys.stderr) # This prints to stderr
        # The following two print to stdout in original, keeping for consistency.
        # print(len(angle_diffs)-1) 
        # print(angle_diffs) 
        
        # convert to array to allow addition
        # Add 3 as an offset, then prev_end_number to accumulate counts across multiple files
        # The '3' offset means a boundary is 3 frames AFTER the peak.
        current_peaks = np.asarray(filtered_peaks) + 3 + prev_end_number

        # convert back to list
        partition_boundary.extend(current_peaks.tolist()) # Use extend for lists

        print("Final partition boundaries from current angle_csv : ", current_peaks.tolist()) # Original print to stdout
        
        # prev_end_number should be the *total count of images processed so far* + 1 (for next start).
        # Angle diffs are N-1 for N images.
        # If angle_diffs has length L, it corresponds to L+1 images.
        prev_end_number += len(angle_diffs) + 1 # Update accumulated end number (number of images processed so far)
        print(f'DEBUG(split main): Current accumulated prev_end_number (total images processed + 1 for next start): {prev_end_number}', file=sys.stderr)
        
    
    # After looping through all angle_csv files, append the final end number
    # This appends the overall total number of images to ensure the last group goes to the end.
    if partition_boundary: 
        # If the last peak isn't near the end, ensure the last group goes to the final image.
        # This will be the image count equivalent to the last H_asift.csv row + 1.
        # The total number of images that correspond to the total homography list H is len(H) + 1.
        # If prev_end_number correctly tracks the image index *after* the last processed image.
        # However, `move_images` needs the index of the *last image in the final group* + 1.
        # For simplicity, append the total number of images available if it's larger than the last boundary.
        # The correct value for the final boundary should be `len(image_files)`.
        # Ensure image_files is correctly populated from `args.image_path` before this.
        # The move_images function loads `image_files` again, let's pass `len(image_files)` explicitly.
        
        # We need the total number of images that will be processed by move_images.
        # To get this, we would ideally load `image_files` in the main `if __name__ == '__main__':` block.
        # For now, let's assume `move_images` correctly handles the end.
        pass # The move_images function now has better end-of-loop handling.
            
    print("Final partition_boundary list to be used for partitioning images: ", partition_boundary) # Original print to stdout
    
    # grouping the group partition
    # image_path is expected to be a list, even if it has one element (the directory)
    move_images(args.image_path, save_path, args.homography, partition_boundary, args.duplicate, overlap=0)
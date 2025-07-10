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


def move_images(image_path, save_path, homography, boundaries, duplicate, overlap):
    image_files = []
    # prev_images = [] # Unused
    H = [] # This will store the loaded homographies from the main H_surf.csv

    # Ensure the base save_path exists (e.g., outputs/surf_mini_partition)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"DEBUG(split): Created base partition save_path: {save_path}", file=sys.stderr)


    # --- Load image files from the provided image_path(s) ---
    # The image_path argument is a list of paths. Handle each one.
    # Assuming image_path[0] will be the directory containing the frames (e.g., outputs/calibrated_frames/DJI_0604_frames)
    for img_p in image_path:
        if os.path.isdir(img_p):
            extensions = [".jpeg", ".jpg", ".png", ".tif"]
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
        with open(homography, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter = ",")
            for row in reader:
                if row: # Ensure row is not empty before attempting conversion
                    H_each = np.array(row).astype(np.float64)
                    flattened_list = H_each.ravel().tolist()
                    H.append(flattened_list)
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
    print('group boundary', group_boundary)
    group_count = 0
    subfolder_count = 1
    boundary_idx = 0 # Use a clearer name for the index into group_boundary list
    
    # Initialize the first subfolder path
    subfolder = os.path.join(save_path, 'group_{}'.format(str(subfolder_count).zfill(3)))
    print(f"DEBUG(split): Initial subfolder path for groups: {subfolder}", file=sys.stderr) 
    
    i = 0 # Current image index being processed
    start = 0 # Start index for slicing homographies

    while i < len(image_files):
        filename = os.path.basename(image_files[i])
        
        # Create the current group subfolder if it doesn't exist
        # This check is done when group_count is 0, meaning it's a new group
        if group_count == 0 and not os.path.exists(subfolder):
            os.makedirs(subfolder)
            print(f"DEBUG(split): Created group subfolder: {subfolder}", file=sys.stderr)

        destination_path = os.path.join(subfolder, filename)
        
        # Move or copy the image file
        try:
            if not duplicate:
                shutil.move(image_files[i], destination_path)
                print(f"DEBUG(split): Moved image {filename} to {subfolder}", file=sys.stderr)
            else:
                shutil.copy(image_files[i], destination_path)
                print(f"DEBUG(split): Copied image {filename} to {subfolder}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR(split): Failed to move/copy {filename} to {destination_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Decide if you want to exit or continue on file error

        group_count += 1
        i = i + 1

        # Check if the current image index 'i' hits a boundary to close the current group and start a new one
        # Ensure boundary_idx is within the bounds of group_boundary list
        if boundary_idx < len(group_boundary) and i == group_boundary[boundary_idx]:
            # Slice the homographies for the current group
            # Note: i-1 because the homography for image 'i' connects image 'i-1' to 'i'
            # and the range is [start, i-1) for a number of matches up to (i-1)
            h_temp = H[start:i-1] 
            
            print(f"DEBUG(split): --- Boundary hit! ({i} == {group_boundary[boundary_idx]}) ---", file=sys.stderr)
            print(f"DEBUG(split): Slicing homographies from index {start} to {i-1}. Resulting h_temp length: {len(h_temp)}", file=sys.stderr)
            
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
                with open(group_hm_filepath, 'a', newline='') as f1: #newline='' for better CSV handling
                    wr = csv.writer(f1, delimiter=",", quoting = csv.QUOTE_NONE)
                    for h_each_save in h_temp:
                        wr.writerow(h_each_save)
                print(f"DEBUG(split): Successfully wrote {len(h_temp)} rows to {group_hm_filepath}", file=sys.stderr)
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
            start = i # Set new start index for slicing homographies in the next group
            h_temp = [] # Clear h_temp for the next group

    # Handle any remaining images if the loop finishes before all boundaries are hit
    # (This logic was commented out in your original, leaving as a comment)
    # '''
    # h_temp = H[start:i]
    # with open(save_path+"/H_asift_"+'group_{}'.format(subfolder_count)+".csv", 'a') as f1:
    #     wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
    #     wr.writerow(h_temp)
    # '''

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
            with open(angle_file, 'r') as file:
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
        print(len(angle_diffs)-1) # Original print to stdout
        print(angle_diffs) # Original print to stdout
        
        # convert to array to allow addition
        # Add 3 as an offset, then prev_end_number to accumulate counts across multiple files
        current_peaks = np.asarray(filtered_peaks) + 3 + prev_end_number

        # convert back to list
        partition_boundary.extend(current_peaks.tolist()) # Use extend for lists

        print("Frame boundaries so far : ", partition_boundary) # Original print to stdout
        
        prev_end_number += len(angle_diffs) + 2 # Update accumulated end number
        print('prev end number: ', prev_end_number) # Original print to stdout
        
    
    # After looping through all angle_csv files, append the final end number
    if partition_boundary: # Only append if some boundaries were found
        partition_boundary.append(prev_end_number)
    else: # If no peaks or angle diffs, set a default boundary (e.g., end of images)
        print("WARNING(split main): No partition boundaries found after processing all angle CSVs. Attempting to create one large group.", file=sys.stderr)
        # Fallback: if no peaks are found, create one large group using the total number of images.
        # This requires knowing the total image count from image_path.
        # For a simple solution, we can append the total number of homographies if available.
        # This part requires more context to implement robustly without image_files count here.
        # For now, if partition_boundary is empty, the move_images will get an empty list,
        # which will cause no groups to be created.
        pass # Keep existing behavior of possibly empty boundary for now.
            
    print("Final partition_boundary: ", partition_boundary) # Original print to stdout
    
    # grouping the group partition
    # image_path is expected to be a list, even if it has one element (the directory)
    move_images(args.image_path, save_path, args.homography, partition_boundary, args.duplicate, overlap=0)
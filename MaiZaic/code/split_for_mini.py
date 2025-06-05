#!/usr/bin/env python

import os
import argparse
# import logging # Not used in the provided snippet
from datetime import datetime
import cv2 # Not used in the provided snippet
import csv
# import stitcher # This import might cause issues if stitcher.py has syntax errors or for circular dependencies. Not directly used in move_images.
# from numpy import genfromtxt # Not used in the provided snippet
import numpy as np
# from numpy.linalg import inv # Not used in the provided snippet
import time # Not used in the provided snippet
import sys
# import copy # Not used in the provided snippet
import shutil
from scipy.signal import find_peaks, medfilt # Used if angle_csv logic is active
import matplotlib # Used if angle_csv logic is active
matplotlib.use('agg') # Use 'agg' backend for non-interactive plotting
import matplotlib.pyplot as plt # Used if angle_csv logic is active
np.set_printoptions(threshold=sys.maxsize)


def move_images(image_path_list, save_path, homography_csv, boundaries, overlap):
    image_files = []
    H = [] # All homographies read from the main H_surf.csv
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print "DEBUG (move_images): Processing image_path(s):", image_path_list
    for path_entry in image_path_list: # Loop through the list of input paths
        if os.path.isdir(path_entry):
            extensions = [".jpeg", ".jpg", ".png"]
            for file_path in sorted(os.listdir(path_entry)):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    image_files.append(os.path.join(path_entry, file_path))
        elif os.path.isfile(path_entry): # If it's a direct file path
            image_files.append(path_entry)
        else:
            print "Warning: %s is not a valid image path or directory." % path_entry

    print "DEBUG (move_images): Total image files found:", len(image_files)

    print "DEBUG (move_images): Attempting to open homography file:", homography_csv
    try:
        with open(homography_csv, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter = ",")
            for row in reader:
                if row:
                    H_each = np.array(row).astype(np.float64)
                    H.append(H_each.tolist()) 
            print "DEBUG (move_images): Total homographies read into H:", len(H)
    except IOError as e: 
        print >> sys.stderr, "ERROR (move_images): Failed to open or read homography file %s: %s" % (homography_csv, e)
        return 
    except Exception as e:
        print >> sys.stderr, "ERROR (move_images): An unexpected error occurred while reading homography file %s: %s" % (homography_csv, e)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return

            
    group_boundary = boundaries
    print 'DEBUG (move_images): Using group boundaries:', group_boundary

    subfolder_count = 1
    boundary_index = 0 
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    i = 0 # Current image file index being processed (0-indexed from image_files list)
    start_index_H = 0 # Start index for slicing the H list for the current group's homographies
    
    while i < len(image_files):
        filename = os.path.basename(image_files[i])
        current_subfolder_path = os.path.join(save_path, 'group_{}'.format(str(subfolder_count).zfill(3)))
        destination_path = os.path.join(current_subfolder_path, filename)
        
        if not os.path.exists(current_subfolder_path): 
            os.makedirs(current_subfolder_path)
            print "DEBUG (move_images): Created subfolder:", current_subfolder_path

        try:
            shutil.copy(image_files[i], destination_path)
            print "DEBUG (move_images): Copied %s to %s" % (filename, current_subfolder_path)
        except Exception as e:
            print >> sys.stderr, "ERROR (move_images): Failed to copy %s to %s: %s" % (image_files[i], destination_path, e)
            return

        # 'i' is the 0-indexed image number.
        # If group_boundary[boundary_index] is B, it means image B is the last of this group.
        # This group contains images with indices from 'first_image_in_group_idx' to 'i'.
        # The number of images is i - first_image_in_group_idx + 1.
        # The number of homographies is (number of images) - 1.
        # Homographies H[first_H_idx_for_group] to H[i-1] are needed. Slice is H[first_H_idx_for_group : i].
        if boundary_index < len(group_boundary) and i == group_boundary[boundary_index]:
            
            h_temp = H[start_index_H : i] 
            
            print "DEBUG (move_images): Boundary for group {} hit at image index i={}. Slicing H from H_index={} to H_index={}. Expected h_temp length: {}. Actual h_temp length: {}".format(
                subfolder_count, i, start_index_H, i, i - start_index_H, len(h_temp)
            )

            target_csv_path = os.path.join(save_path, "H_asift_group_{}.csv".format(str(subfolder_count).zfill(3)))
            print "DEBUG (move_images): Attempting to write {} homographies to: {}".format(len(h_temp), target_csv_path)
            try:
                with open(target_csv_path, 'w') as f1: 
                    wr = csv.writer(f1, delimiter=",", quoting=csv.QUOTE_NONE)
                    for h_each_save in h_temp:
                        wr.writerow(h_each_save)
                print "DEBUG (move_images): Successfully wrote group {} homographies.".format(subfolder_count)
            except Exception as e:
                print >> sys.stderr, "ERROR (move_images): Failed to write group {} homographies to CSV: {}".format(subfolder_count, e)
                return 

            subfolder_count += 1
            boundary_index += 1 
            start_index_H = i # Next group's homographies start from H[i]
            
            if overlap > 0 and boundary_index < len(group_boundary): # Only step back if there's overlap and more groups
                 print "DEBUG (move_images): Overlap > 0, stepping image index 'i' back by {} from {} for next group.".format(overlap, i)
                 i = i - overlap 
                 # Note: if i becomes negative, ensure loop condition `i < len(image_files)` and list access handle it.
                 # For overlap, typically you ensure `i - overlap` is still >= 0 or handle the start of the sequence.

        i += 1 

    # After the image loop, handle any remaining homographies for the last segment of images
    if start_index_H < len(H) and start_index_H < (len(image_files) -1) :
        # This segment covers images from index start_index_H up to len(image_files)-1.
        # The homographies needed are H[start_index_H] through H[len(image_files)-2].
        # So the slice for H is H[start_index_H : len(image_files)-1].
        
        end_slice_H_for_last_group = len(image_files) - 1 # H index for pair (N-2, N-1) is N-2. Slice goes up to N-1.
        
        h_temp_last = []
        if start_index_H < end_slice_H_for_last_group: # Ensure there's at least one homography
             h_temp_last = H[start_index_H : end_slice_H_for_last_group]
        
        if len(h_temp_last) > 0:
            target_csv_path = os.path.join(save_path, "H_asift_group_{}.csv".format(str(subfolder_count).zfill(3)))
            print "DEBUG (move_images): Writing final segment ({} homographies, from H_index {} to {}) to: {}".format(
                len(h_temp_last), start_index_H, start_index_H + len(h_temp_last) -1, target_csv_path
            )
            try:
                with open(target_csv_path, 'w') as f1: 
                    wr = csv.writer(f1, delimiter=",", quoting=csv.QUOTE_NONE)
                    for h_each_save in h_temp_last:
                        wr.writerow(h_each_save)
                print "DEBUG (move_images): Final segment homographies written for group {}.".format(subfolder_count)
            except Exception as e:
                print >> sys.stderr, "ERROR (move_images): Failed to write final segment homographies to CSV: {}".format(e)

def filter_on_shoulder(peaks, properties, data, window_size):
    filtered_peaks = []
    for peak in peaks:
        start = max(0, peak - window_size)
        end = min(len(data), peak + window_size)
        # print peak, start, end # Python 2.7 print statement
        if data[peak] == max(data[start:end]):
            filtered_peaks.append(peak)
    return filtered_peaks

def median_filter_smoothing(data, window_size):
    return medfilt(data, window_size)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / float(window_size), mode='valid')

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
    if len(filtered_peaks) > 0: # Ensure filtered_peaks is not empty before indexing angle_diffs
        plt.plot(filtered_peaks, angle_diffs[np.array(filtered_peaks)], "x", label='peaks', color = 'red')
    else:
        print "Warning (plot_partition): No peaks to plot."
    plt.title("data with detected peaks")
    plt.xlabel("frame number")
    plt.ylabel("angle difference")
    plt.legend()
    plt.savefig(save_path)
    plt.close() # Close plot to free memory

# ... (imports and function definitions: move_images, filter_on_shoulder, etc. from previous version) ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', required=True, help="paths to one or more images or image directories")
    parser.add_argument("-save_path", dest='save_path', default="RESULTS/global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
    parser.add_argument('-hm', '--homography', type=str, required=True, help='txt file that stores homography matrices')
    parser.add_argument('-angle_csv', '--angle_csv', type=str, nargs='*', help='csv file(s) that store the angle difference(s)')

    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    partition_boundary = []
    
    # --- Determine total number of images first (this is now at a higher scope) ---
    num_total_images = 0
    temp_image_files = [] 
    if args.image_path:
        for path_entry in args.image_path:
            if os.path.isdir(path_entry):
                extensions = [".jpeg", ".jpg", ".png"]
                try:
                    for file_path in sorted(os.listdir(path_entry)):
                        if os.path.splitext(file_path)[1].lower() in extensions:
                            temp_image_files.append(os.path.join(path_entry, file_path))
                except OSError as e:
                    print "Warning: Could not read directory {} while counting images: {}".format(path_entry, e)
            elif os.path.isfile(path_entry): # If it's a direct file path
                temp_image_files.append(path_entry)
    num_total_images = len(temp_image_files)

    if num_total_images == 0:
        print "Error: No image files found from -image_path. Cannot proceed."
        sys.exit(1)
    print "INFO: Total image files found for processing: {}".format(num_total_images)
    # --- End of determining total number of images ---

    if args.angle_csv and len(args.angle_csv) > 0:
        print "INFO: Using dynamic partition boundary detection from -angle_csv file(s): {}".format(", ".join(args.angle_csv))
        
        all_angle_diffs_combined = []
        # Correctly iterate if args.angle_csv is a list of files
        angle_files_to_process = args.angle_csv
        if not isinstance(args.angle_csv, list): # Handle if nargs='*' makes it a single item list or None
            angle_files_to_process = [args.angle_csv] if args.angle_csv else []

        for angle_file in angle_files_to_process:
            if not angle_file: continue # Skip if an empty string or None was in the list
            current_file_angle_diffs = []
            try:
                with open(angle_file, 'r') as file_handle:
                    angle_reader = csv.reader(file_handle)
                    for row in angle_reader:
                        if row: 
                            current_file_angle_diffs.append(float(row[0]))
                all_angle_diffs_combined.extend(current_file_angle_diffs)
                print "Read {} angle differences from {}.".format(len(current_file_angle_diffs), angle_file)
            except IOError as e:
                print "Error reading angle_csv file {}: {}".format(angle_file, e)
            except ValueError as e:
                print "Error converting value to float in {}: {}".format(angle_file, e)

        if not all_angle_diffs_combined:
            print "Warning: No data read from any angle_csv file(s). Falling back to manual/default partitioning."
            # Ensure partition_boundary remains empty to trigger fallback
            partition_boundary = [] 
        else:
            angle_diffs_np = np.asarray(all_angle_diffs_combined)
            
            angle_threshold_factor = 1.0 
            min_peak_distance_abs = 5 # Absolute minimum distance between peaks (in terms of frames)
            # Relative distance, e.g., 5% of sequence length, but not less than absolute min
            min_peak_distance_rel = max(1, int(len(angle_diffs_np) * 0.05)) 
            min_peak_distance = max(min_peak_distance_abs, min_peak_distance_rel)
            
            if len(angle_diffs_np) > 1 :
                dynamic_thresh = np.mean(angle_diffs_np) + angle_threshold_factor * np.std(angle_diffs_np)
                print "Dynamic threshold for peak detection: {:.2f}".format(dynamic_thresh)

                safe_peak_distance = min(min_peak_distance, len(angle_diffs_np) -1 if len(angle_diffs_np) > 1 else 1)
                if safe_peak_distance <=0: safe_peak_distance = 1
                
                filtered_peaks_indices, properties = find_peaks(angle_diffs_np, height=dynamic_thresh, distance=safe_peak_distance)
                
                # The filtered_peaks_indices are 0-indexed relative to angle_diffs_np.
                # If angle_diffs_np[j] is the diff between image j and image j+1,
                # then a peak at index j means image j (0-indexed in the sequence of images
                # that produced the diffs) is a boundary.
                partition_boundary = sorted(list(set(filtered_peaks_indices.tolist())))

                plot_save_path = os.path.join(save_path, 'angle_peaks_plot.png')
                plot_partition(angle_diffs_np, filtered_peaks_indices, plot_save_path) 
                print "Saved angle peaks plot to:", plot_save_path
                print "Dynamically determined raw boundary frame indices: {}".format(partition_boundary)
            else:
                print "Warning: Not enough data points in angle_diffs (need >1) for dynamic peak detection."
                partition_boundary = [] # Ensure fallback if this path is taken
    
    if not partition_boundary: # Fallback or Manual Partitioning
        if args.angle_csv and len(args.angle_csv) > 0:
             print "Warning: Dynamic boundary detection failed to find any peaks. Falling back to manual/default partitioning for {} images.".format(num_total_images)
        else:
            print "INFO: -angle_csv not provided or empty. Using manual/default partition boundary for {} images.".format(num_total_images)

        if num_total_images == 117: 
            partition_boundary = [29, 59, 89, 116] 
            # Example: partition_boundary = [14, 29, 44, 59, 74, 89, 104, 116] # For smaller groups
            print "Using specific manual partition boundary for 117 images: ", partition_boundary
        elif num_total_images > 0:
            images_per_group_target = 30 
            num_groups = max(1, int(np.ceil(float(num_total_images) / images_per_group_target)))
            temp_boundaries = []
            current_boundary = -1
            for k_group in range(num_groups):
                # The boundary is the index of the last image IN THE GROUP.
                # Group k images: from current_boundary + 1 to min(current_boundary + images_per_group_target, num_total_images -1)
                next_boundary_candidate = current_boundary + images_per_group_target
                boundary = min(next_boundary_candidate, num_total_images - 1)
                temp_boundaries.append(boundary)
                current_boundary = boundary
                if boundary == num_total_images - 1:
                    break
            partition_boundary = sorted(list(set(temp_boundaries)))
            print "Using default partitioning into {} groups. Boundaries: {}".format(len(partition_boundary), partition_boundary)
        # else: # num_total_images == 0, already handled by exit at the top

    # --- Final check and adjustment of partition_boundary ---
    # num_total_images is now defined globally in this __main__ block
    if not partition_boundary and num_total_images > 0: 
        partition_boundary = [num_total_images - 1]
        print "Fallback: Creating a single group for all {} images as no boundaries were defined.".format(num_total_images)
    elif num_total_images > 0 and partition_boundary: # Ensure partition_boundary is not empty
        # Filter out boundaries beyond the actual number of images
        partition_boundary = [b for b in partition_boundary if b < num_total_images] 
        # Ensure the last image is always a boundary
        if not partition_boundary or partition_boundary[-1] < num_total_images - 1:
            if partition_boundary and partition_boundary[-1] < num_total_images - 1 :
                 partition_boundary.append(num_total_images - 1)
            elif not partition_boundary : 
                 partition_boundary = [num_total_images-1]
        
        partition_boundary = sorted(list(set(partition_boundary))) 
        print "Final adjusted partition boundaries: {}".format(partition_boundary)
    elif num_total_images == 0: # Should have already exited
        print "Critical Error: No images found, cannot define partition boundaries."
        sys.exit(1)
    # If partition_boundary is still empty after all this, something is wrong.
    if not partition_boundary:
        print "CRITICAL ERROR: partition_boundary is empty after all checks. Exiting."
        sys.exit(1)
            
    move_images(args.image_path, save_path, args.homography, partition_boundary, overlap = 0)
#!/usr/bin/env python

'''
author: Dewi Kharismawati

this experiment of mosaicking with surf:
    - estimating homograhy matrices
    - then saving to the csv file in desired directory
'''
import os
import argparse
import logging
from datetime import datetime
import cv2
import csv
import numpy as np
from numpy.linalg import inv
from surf import surf
import time
import sys
import copy # Not directly used in the provided snippet but kept for completeness
import shutil # Not directly used in the provided snippet but kept for completeness
from scipy.signal import find_peaks, medfilt # Not directly used in this script, but in split_for_mini
import matplotlib # Not directly used in this script
matplotlib.use('agg') # For plotting
import matplotlib.pyplot as plt # Not directly used in this script
np.set_printoptions(threshold=sys.maxsize) # For print formatting
import glob # <--- ADDED IMPORT for robust file listing
import traceback # <--- ADDED IMPORT for error printing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, help="path to directory containing images") # <--- Modified nargs='+' to type=str for single directory input
    parser.add_argument('-hm', '--homography', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-save_path', dest='save_path', default="homography_matrices/", type=str, help="path to save result")
    parser.add_argument('-scale', '--scale', dest='scale', default=1, type=int, help="size down scale ratio")
    args = parser.parse_args()

    scale = args.scale
    
    # These variables seem unused but are kept as per your request for completeness
    result = None
    result_gry = None

    frames_to_mosaic_path = args.save_path + '/frames_to_mosaic/'
    hm_path = args.save_path + '/homography_matrices/'

    if not os.path.exists(frames_to_mosaic_path):
        os.makedirs(frames_to_mosaic_path)

    if not os.path.exists(hm_path):
        os.makedirs(hm_path)

    # --- START OF MODIFIED IMAGE LOADING SECTION (for robust directory handling) ---
    input_image_dir = args.image_path
    # Assuming TIFFs are the correct input format as per previous discussions
    image_pattern = os.path.join(input_image_dir, '*.tif') 
    all_image_files = sorted(glob.glob(image_pattern))

    print("DEBUG: Input image directory: {}".format(input_image_dir), file=sys.stderr)
    print("DEBUG: Glob pattern used: {}".format(image_pattern), file=sys.stderr)
    print("DEBUG: Number of image files found for processing: {}".format(len(all_image_files)), file=sys.stderr)

    if not all_image_files:
        print("ERROR: No image files found in the specified input path. Please check -image_path and file types. Exiting.", file=sys.stderr)
        sys.exit(1) # Exit if no images
    print("DEBUG: First image for processing: {}".format(all_image_files[0]), file=sys.stderr)
    # --- END OF MODIFIED IMAGE LOADING SECTION ---

    H_tp = np.array([[0,0,0],[0,0,0],[0,0,0]]) # Unused, but kept for completeness
    H_flat = None # Initialize H_flat for scope
    
    image_index = -1
    prev_gray = None # Initialize prev_gray outside the loop
    successful_homography_writes = 0 # New counter for successful writes

    for current_image_path in all_image_files: # Iterate through the clean list of files
        print("DEBUG: Currently processing file: {}".format(current_image_path), file=sys.stderr) # <--- New debug print

        image_color_big = cv2.imread(current_image_path)
        
        if image_color_big is None:
            print("WARNING: Could not read image {}. Skipping.".format(current_image_path), file=sys.stderr)
            continue # Skip to next image if read fails

        filename = os.path.basename(current_image_path)
        height, width, channel = image_color_big.shape
        sw = int(width/scale) # Use scale variable
        sh = int(height/scale) # Use scale variable

        image_color = cv2.resize(image_color_big, (sw,sh))

        print(filename) # Original print

        # Save processed frame to frames_to_mosaic_path
        cv2.imwrite(os.path.join(frames_to_mosaic_path,filename), image_color)
        
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
        
        image_index += 1

        if image_index == 0:
            prev_gray = image_gray
            continue # Skip homography for the first image

        print("counter {}".format(image_index), file=sys.stderr) # Original print, corrected to image_index for counter

        h_time = time.time()
        current_H = surf(prev_gray, image_gray) # Call the surf function
        elapsed_time_h = time.time()-h_time

        print("DEBUG: Homography H result: {}".format(current_H), file=sys.stderr) # <--- New debug print

        # ONLY proceed if a valid homography matrix was returned
        if current_H is not None:
            print("DEBUG: Homography calculated and is not None for pair {} to {}.".format(image_index-1, image_index), file=sys.stderr)
            try:
                H_flat = np.array(current_H).flatten().astype(np.float64)
                print("Homography Flat: {}".format(H_flat))
                print("Saving to: {}".format(hm_path))
                
                # Write to H_surf.csv
                with open(os.path.join(hm_path, "H_surf.csv"), 'a') as f1:
                    wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
                    wr.writerow(H_flat)
                    f1.flush() # Force flush to disk
                    os.fsync(f1.fileno()) # Force OS flush
                    print("DEBUG: Wrote a row to H_surf.csv for image index {}.".format(image_index), file=sys.stderr)

                # Write to H_surf_time_elapsed.csv
                with open(os.path.join(hm_path, "H_surf_time_elapsed.csv"), 'a') as f2:
                    twr = csv.writer(f2,  delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
                    twr.writerow([elapsed_time_h])
                    f2.flush() # Force flush to disk
                    os.fsync(f2.fileno()) # Force OS flush
                    print("DEBUG: Wrote a row to H_surf_time_elapsed.csv for image index {}.".format(image_index), file=sys.stderr)
                
                successful_homography_writes += 1 # Increment counter on successful write

            except Exception as e:
                print("ERROR: Failed to write homography to CSV for pair {} to {}: {}".format(image_index-1, image_index, e), file=sys.stderr)
                traceback.print_exc(file=sys.stderr) # Print full traceback to stderr
        else:
            print("Skipping homography save for pair {} to {} due to insufficient matches or error.".format(image_index-1, image_index), file=sys.stderr)
        
        prev_gray = image_gray # Update prev_gray for the next iteration
        
    print("DEBUG: Total successful homography rows written (reported by script): {}".format(successful_homography_writes), file=sys.stderr) # Final count
    print("Homography matrices have been processed and saved to " + hm_path)
#!/usr/bin/env python

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-hm', '--homography', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-save_path', dest='save_path', default="homography_matrices/", type=str, help="path to save result")
    parser.add_argument('-scale', '--scale', dest='scale', default=1, type=int, help="size down scale ratio")
    args = parser.parse_args()

    scale = args.scale # Define 'scale' variable
    
    result = None # Unused
    result_gry = None # Unused

    frames_to_mosaic_path = args.save_path + '/frames_to_mosaic/'
    hm_path = args.save_path + '/homography_matrices/'

    if not os.path.exists(frames_to_mosaic_path):
            os.makedirs(frames_to_mosaic_path)

    if not os.path.exists(hm_path):
            os.makedirs(hm_path)

    # Ensure to get a flat list of all image files
    all_image_files = []
    for p in args.image_path:
        if os.path.isdir(p):
            extensions = [".jpeg", ".jpg", ".png"]
            for file_path in sorted(os.listdir(p)):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    all_image_files.append(os.path.join(p, file_path))
        elif os.path.isfile(p):
            all_image_files.append(p)
        else:
            print "Warning: %s is not a valid path or directory." % p
            continue
    
    all_image_files.sort() # Sort the image paths for consistent pairing

    H_tp = np.array([[0,0,0],[0,0,0],[0,0,0]]) # Unused, but kept for completeness

    image_index = -1
    prev_gray = None # Initialize prev_gray outside the loop
    successful_homography_writes = 0 # New counter for successful writes

    for image_path in all_image_files:
        print "reading current frame from {0}".format(image_path)
        image_color_big = cv2.imread(image_path)
        
        if image_color_big is None:
            print "Error: Could not read image {0}. Skipping.".format(image_path)
            continue

        filename = os.path.basename(image_path)
        height, width, channel = image_color_big.shape
        sw = int(width/scale)
        sh = int(height/scale)

        image_color = cv2.resize(image_color_big, (sw,sh))

        print filename

        # Save processed frame to frames_to_mosaic_path
        cv2.imwrite(os.path.join(frames_to_mosaic_path,filename), image_color)
        
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
        
        image_index += 1

        if image_index == 0:
            prev_gray = image_gray
            continue

        print "counter ", image_index

        h_time = time.time()
        current_H = surf(prev_gray, image_gray) # Call the surf function
        elapsed_time_h = time.time()-h_time

        # ONLY proceed if a valid homography matrix was returned
        if current_H is not None:
            print >> sys.stderr, "DEBUG: Homography calculated and is not None for pair {0} to {1}.".format(image_index-1, image_index)
            try:
                H_flat = np.array(current_H).flatten().astype(np.float64)
                print "Homography Flat:", H_flat
                print "Saving to:", hm_path
                
                # Write to H_surf.csv
                with open(hm_path+"/H_surf.csv", 'a') as f1:
                    wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
                    wr.writerow(H_flat)
                    f1.flush() # Force flush to disk
                    os.fsync(f1.fileno()) # Force OS flush
                    print >> sys.stderr, "DEBUG: Wrote a row to H_surf.csv for image index {0}.".format(image_index)

                # Write to H_surf_time_elapsed.csv
                with open(hm_path+"/H_surf_time_elapsed.csv", 'a') as f2:
                    twr = csv.writer(f2,  delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
                    twr.writerow([elapsed_time_h])
                    f2.flush() # Force flush to disk
                    os.fsync(f2.fileno()) # Force OS flush
                    print >> sys.stderr, "DEBUG: Wrote a row to H_surf_time_elapsed.csv for image index {0}.".format(image_index)
                
                successful_homography_writes += 1 # Increment counter on successful write

            except Exception as e:
                print >> sys.stderr, "ERROR: Failed to write homography to CSV for pair {0} to {1}: {2}".format(image_index-1, image_index, e)
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            print "Skipping homography save for pair {0} to {1} due to insufficient matches or error.".format(image_index-1, image_index)
        
        prev_gray = image_gray # Update prev_gray for the next iteration
        
    print "DEBUG: Total successful homography rows written (reported by script):", successful_homography_writes # Final count
    print "Homography matrices have been processed and saved to " + hm_path
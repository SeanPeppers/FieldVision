#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
project: cornetv2

author: Dewi Kharismawati

aboutL:
    - this is asift_mosaic_final.py
    - this will compute homography between frames using asift
    - and homography matrix 1x9 will be save into the csv file

call:

    python asift_mosaic_final -image_path /path/to/raw/image -hm /path/to/homography/csv/file -save_path /path/to/save/path


'''

import os
import argparse
import logging
from datetime import datetime
import cv2
import csv
import sys # Import sys for sys.exit
# Corrected to import 'stitcher' module, assuming stitcher.py is directly under /app/code/
from src import stitcher 
from numpy import genfromtxt
import numpy as np
from numpy.linalg import inv
# Corrected to use absolute import path from the 'code' package
from src.asift.asift import my_asift 
import time
# Removed redundant 'import csv' as it's already imported above


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-hm', '--homography', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-b', '--debug', dest='debug', action='store_true', help='enable debug logging')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='disable all logging')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help="display result")
    parser.add_argument('-s', '--save', dest='save', action='store_true', help="save result to file")
    parser.add_argument('-save_path', dest='save_path', default="results/stitched_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
    parser.add_argument('-k', '--knn', dest='knn', default=2, type=int, help="Knn cluster value")
    parser.add_argument('-l', '--lowe', dest='lowe', default=0.7, type=float, help='acceptable distance between points')
    parser.add_argument('-scale', '--scale', dest='scale', default=1, type=int, help='downsampling ratio for images')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    logging.info("beginning sequential matching")

    frames_to_mosaic_path = args.save_path + '/frames_to_mosaic/'
    hm_path = args.save_path + '/homography_matrices/'

    if not os.path.exists(frames_to_mosaic_path):
         os.makedirs(frames_to_mosaic_path)
        
    if not os.path.exists(hm_path):
         os.makedirs(hm_path)

    # Refactored image path collection for robustness
    input_paths = args.image_path # This is a list from argparse
    all_image_files = []

    for path in input_paths:
        if not os.path.exists(path):
            logger.error(f"Input path '{path}' does not exist. Skipping.")
            continue
        if os.path.isdir(path):
            # Added .tif/.tiff extensions as they are common for aerial imagery
            extensions = [".jpeg", ".jpg", ".png", ".JPG", ".tif", ".tiff"] 
            for file_name in sorted(os.listdir(path)):
                if os.path.splitext(file_name)[1].lower() in extensions:
                    all_image_files.append(os.path.join(path, file_name))
        else: # It's a single file
            all_image_files.append(path)

    image_paths = sorted(all_image_files) # Ensure consistent order for processing
    logger.info(f"Found {len(image_paths)} images to process for homography estimation.")

    if not image_paths:
        logger.error("No images found for processing. Exiting.")
        sys.exit(1) # Exit if no images

    result = None
    result_gry = None

    homography = args.homography
    image_index = -1
    counter = 0
    # Renamed H to H_list to avoid conflict with local H_matrix variable in the loop
    H_list = [] 
    points_in = np.array([[0,0], [0,0],[0,0],[0,0]], dtype=np.float32)
    H_tp = np.array([[0,0,0],[0,0,0],[0,0,0]])
    

    # Looping through the pre-processed list of image files
    for current_image_path in image_paths: 

        logging.info(f"Reading image from {current_image_path}")
        image_color_big = cv2.imread(current_image_path)
        
        if image_color_big is None:
            logger.error(f"Failed to load image: {current_image_path}. Skipping this image.")
            continue

        filename = os.path.basename(current_image_path)
        height, width, channel = image_color_big.shape
        sw = int(width/args.scale)
        sh = int(height/args.scale)

        image_color = cv2.resize(image_color_big, (sw,sh))

        print(f"Processing: {filename}") 

        cv2.imwrite(os.path.join(frames_to_mosaic_path,filename), image_color)
        
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
        
        image_index += 1

        if image_index == 0:
            prev_gray = image_gray
            # No homography to compute for the first image, just set it as previous
            continue

        print(f"Counter: {counter}") 

        h_time = time.time()
        # Renamed H to H_matrix to avoid shadowing the H_list variable
        H_matrix = my_asift(prev_gray, image_gray) 
        elapsed_time_h = time.time()-h_time

        # Check if H_matrix is None, which indicates insufficient matches for homography estimation
        if H_matrix is None:
            logger.warning(f"Homography estimation failed for image pair (previous image, {filename}). No homography matrix will be saved for this pair.")
            prev_gray = image_gray # Still update prev_gray to try matching with the next image
            continue # Skip to the next image if homography couldn't be computed

        H_flat = np.array(H_matrix).flatten().astype(np.float64)
        print(f"Computed Homography (flat): {H_flat}") 
        print(f"Saving to: {args.save_path}")        
        
        with open(hm_path +"/H_asift.csv", 'a') as f1:
           wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
           wr.writerow(H_flat)

        with open(hm_path+"/H_asift_time_elapsed.csv", 'a') as f2:
           twr = csv.writer(f2,  delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
           twr.writerow([elapsed_time_h])
         
        counter+=1
        prev_gray = image_gray
    logger.info("processing complete!")
        
    # The following lines are commented out as they were in your original file.
    # They refer to functions like 'stitcher.combine_images', 'stitcher.helpers.display',
    # and 'stitcher.helpers.save_image'.
    # If 'stitcher' (now imported) provides these functions, you would need to adjust the calls.
    # For example, if stitcher.py had a 'combine_images' function, it would be:
    # result, H_tp = stitcher.combine_images(image_color, result, counter, H, H_tp)

    # if args.display and not args.quiet:
    #     stitcher.helpers.display('result', result) # Assuming stitcher has a 'helpers' submodule
    #     if cv2.waitKey(200) & 0xFF == ord('q'):
    #         pass # Use pass instead of break if no display is intended
    
    # stitcher.helpers.save_image(args.save_path+"Frame_"+str(counter)+".png", result)
  
    # if args.display and not args.quiet:
    #     cv2.destroyAllWindows()
    # if args.save:
    #     logger.info("saving stitched image to {0}".format(args.save_path))
    #     stitcher.helpers.save_image(args.save_path, result)

#!/usr/bin/env python

# Built-in Modules
import os
import argparse
import logging
from datetime import datetime
import cv2
import csv
import image_stitching # Assuming this is available and handles display if needed
import numpy as np
from numpy.linalg import inv
import time
import sys
import copy
np.set_printoptions(threshold=sys.maxsize)

start = time.time()
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-start', '--start', dest='start', default = 0, type=int, help="start stitching at (currently unused)")
    parser.add_argument('-stop', '--stop', default = 10000, type=int, help="stop stitching at")
    parser.add_argument('-save_path', dest='save_path', default="RESULTS/global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
    parser.add_argument('-hm', '--homography', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-fname', '--fname', dest='fname', default='ASIFT', help='filename')
    #parser.add_argument('-video', '--videos', dest = 'video', type=str, default= 'N', help='save for videos?')
    parser.add_argument('-scale', '--scale', dest='scale', default=1, type=int, help='image size scale')
    parser.add_argument('-mini_mosaic', '--mini_mosaic', dest='mini_mosaic', action='store_true', help='enable mini mosaic')
    

    args = parser.parse_args()

    save_path = args.save_path
    print(args.mini_mosaic)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    if args.mini_mosaic:
        mini_path = args.save_path
        if not os.path.exists(mini_path):
             os.makedirs(mini_path)

        save_path = mini_path

    result = None
    result_gry = None

    stop = args.stop
    homography_matrix = args.homography
    image_paths = args.image_path
    image_index = 0 # Initialize image_index to 0
    counter = 0
    
    H_cum = [] 
    H=[]
    cor2 = []
    corners_h = [] 
    temp_c_normalized = np.zeros((3,4)) # Unused, can be removed
    
    with open(homography_matrix, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter = ",")

        for row in reader:
            H_each = np.asarray(row, dtype=np.float).reshape(3,3)

            H.append(H_each)

            if counter == 0:
               # H[0] is H_img0_to_img1. H_cum[0] should be H_img1_to_img0.
               H_temp = inv(H[counter]) 
               H_cum.append((H_temp))
               
            elif counter > 0:
               # H_cum[counter-1] is H_img_counter_to_img0.
               # H[counter] is H_img_counter_to_img_counter+1.
               # We need H_img_counter+1_to_img0.
               # H_img_counter+1_to_img0 = H_img_counter+1_to_img_counter . H_img_counter_to_img0
               #                        = inv(H[counter]) . H_cum[counter-1]
               H_temp = np.dot(inv(H[counter]), H_cum[counter-1]) # Corrected order for accumulation
               H_cum.append(H_temp)
            
            if counter >= stop: # Changed from == to >= to handle exact stop point
               break
            
            counter = counter+1

    H_cum_new = np.asarray(H_cum)

    max_x, max_y = 0, 0
    min_x, min_y = float('inf'), float('inf')

    # Checking images only once for max_x, min_x, max_y, min_y calculation
    image_paths_list = []
    for image_path in image_paths:
        if os.path.isdir(image_path):
            extensions = [".jpeg", ".jpg", ".png", "JPG"]
            for file_path in sorted(os.listdir(image_path), reverse=False):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    image_paths_list.append(os.path.join(image_path, file_path))
        else:
            image_paths_list.append(image_path)
    
    count = 0
    # Process only images up to the 'stop' limit if H_cum has enough homographies
    num_images_to_consider = min(len(image_paths_list), len(H_cum_new) + 1)

    for i in range(num_images_to_consider): # Iterate through indices
        image_path = image_paths_list[i]
        image_rgb = cv2.imread(image_path)
        if image_rgb is None:
            print("no image found at path: ", image_path)
            continue
        h, w = image_rgb.shape[:2]
        # print(f"Processing image {i}: {h}x{w}") # Debug print

        corners_4 = np.array([[0,0], [w,0],[w,h],[0,h]], dtype=np.float32)
        if i == 0: # First image in the sequence (index 0)
            # Its corners are its own dimensions, relative to itself
            corners_h.append(corners_4.reshape((-1,1,2)))
        else:
            # Transform corners of image 'i' using H_cum_new[i-1] (H_img_i_to_img_0)
            corners_h.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), H_cum_new[i-1]))
        
        # This 'count' variable is redundant as 'i' serves the same purpose
        # and has already been incremented in the loop. Can be removed if needed.
        # count += 1 

    corners_h_arr = np.asarray(corners_h)

    # Filter out any NaN/Inf that might result from bad transformations
    valid_corners = corners_h_arr[np.isfinite(corners_h_arr).all(axis=(1,2))]

    if valid_corners.shape[0] == 0:
        print("Error: No valid transformed corners to calculate canvas size. Using default image 0 size.")
        # Fallback to the first image's size if no valid corners
        first_image_rgb = cv2.imread(image_paths_list[0])
        if first_image_rgb is not None:
            h_ref, w_ref = first_image_rgb.shape[:2]
            min_x, max_x = 0, float(w_ref)
            min_y, max_y = 0, float(h_ref)
        else:
            print("Critical Error: Cannot even read the first image for fallback dimensions. Exiting.")
            sys.exit(1)
    else:
        min_x = np.min(valid_corners[:,0,0])
        max_x = np.max(valid_corners[:,0,0])
        min_y = np.min(valid_corners[:,0,1])
        max_y = np.max(valid_corners[:,0,1])

    print(f"Initial min/max: x_min={min_x:.2f}, x_max={max_x:.2f}, y_min={min_y:.2f}, y_max={max_y:.2f}")

    offset_x = 0.0
    if min_x < 0: # Use < 0 to handle negative coordinates
       offset_x = np.ceil(-min_x)
    
    offset_y = 0.0
    if min_y < 0: # Use < 0 to handle negative coordinates
       offset_y = np.ceil(-min_y)

    # Calculate canvas dimensions including offsets
    canvas_width = int(np.ceil(max_x + offset_x)) - int(np.floor(min_x + offset_x))
    canvas_height = int(np.ceil(max_y + offset_y)) - int(np.floor(min_y + offset_y))

    # Ensure minimum dimensions if somehow calculated as non-positive
    if canvas_width <= 0: canvas_width = cv2.imread(image_paths_list[0]).shape[1] if image_paths_list else 1
    if canvas_height <= 0: canvas_height = cv2.imread(image_paths_list[0]).shape[0] if image_paths_list else 1

    print(f"Offsets: offset_x={offset_x:.2f}, offset_y={offset_y:.2f}")
    print(f"Calculated canvas size (W x H): {canvas_width} x {canvas_height}")

    offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
    print('offset matrix:\n', offset_matrix)
    
    # Corrected global_mosaic initialization: (height, width, channels)
    global_mosaic = np.zeros((canvas_height, canvas_width, 3), np.uint8)
    print('global mosaic shape: ', np.shape(global_mosaic))
    
    # Destination size for warpPerspective is (width, height)
    warp_size = (canvas_width, canvas_height) 

    # 'row' and 'col' are now correctly derived from global_mosaic's shape
    row, col, channel = global_mosaic.shape # row is height, col is width
    
    # This mask is for the initial canvas, and will be updated with each image blend
    # Not used directly in the blending logic below, can be removed if not for initial visual
    # mask = np.ones((col,row), np.uint8) # This would be (width, height) for initial mask
    # mask = mask*255

    image_index = 0 # Reset image_index for the main stitching loop
    for image_path in image_paths_list:
        image_rgb = cv2.imread(image_path)
        if image_rgb is None:
            print("error no image at path: ", image_path)
            continue

        h, w = image_rgb.shape[:2]
        corners_4 = np.array([[1,1], [w,1],[w,h],[1,h]], dtype=np.float32)

        # It's generally not necessary to resize image_rgb to itself (w,h) unless 'scale' is used.
        # If 'scale' is meant to be applied here, it should be adjusted.
        # image_rgb = cv2.resize(image_rgb, (w,h)) 
        
        print(f"Processing image {image_index+1}/{len(image_paths_list)}: {os.path.basename(image_path)}")
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
 
        if image_index == 0:
            print("Warping first image (index 0) into global mosaic.")
            global_mosaic = cv2.warpPerspective(image_rgb, offset_matrix, warp_size)
            cor2.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), offset_matrix))
            
        elif image_index > 0:
            # H_cum_new[image_index-1] is H_img_image_index_to_img_0, which is what we need.
            current_image_transform = np.dot(offset_matrix, H_cum_new[image_index-1])
            wrapped = cv2.warpPerspective(image_rgb, current_image_transform, warp_size)
    
            (ret,data_map) = cv2.threshold(cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
            data_map = cv2.erode(data_map, np.ones((10,10), np.uint8))
            
            # Blend the current warped image onto the global mosaic
            # Ensure proper casting to uint8 for cv2.add if not already.
            temp_mosaic = cv2.add(global_mosaic, 0, mask=np.bitwise_not(data_map))
            wrapped_active = cv2.add(wrapped, 0, mask=data_map)
            global_mosaic = cv2.add(temp_mosaic, wrapped_active)

            cor2.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), current_image_transform))
        
        image_index += 1
        
        # Assuming image_stitching.helpers.display is available
        if 'image_stitching' in sys.modules and hasattr(image_stitching, 'helpers') and hasattr(image_stitching.helpers, 'display'):
            image_stitching.helpers.display('mosaic_global2', global_mosaic)
        else:
            # Fallback for display if the module or function is not found
            # This is commented out as it requires a display environment
            # cv2.imshow('mosaic_global2', global_mosaic)
            pass
        cv2.waitKey(200)

    print(f"Saving final mosaic to: {save_path}")
    final_output_filename = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"_" + args.fname + ".png")
    cv2.imwrite(final_output_filename, global_mosaic)
    print(f"Final mosaic saved as: {final_output_filename}")
    end = time.time()

    print("time elapsed: "+  str(end-start)+  " seconds")
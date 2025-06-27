'''
project: MaiZaic

author: Dewi Kharismawati

about this script:
    - this is mini_mosaic_360.py
    - this is to create a global mosaic for all minimosaic using asift
    - result will be png file of global mosaic
    - homography between mini mosaic also will be save in the save_path


to call:
    python mini_mosaic_360.py -image_path /path/to/mini/mosaic -save_path /path/to/save

'''


import os
import argparse
import cv2
import csv
from numpy import genfromtxt
import numpy as np
from numpy.linalg import inv
from code.asift.asift import my_asift
from datetime import datetime
import time
# Note: csv is imported twice, but not harmful.

# Assuming common.py is in the 'code' directory, not 'code/asift'
from code.asift.common import Timer

# Assuming find_obj.py is in the 'code' directory
# You will need to ensure 'explore_match' signature in find_obj.py matches how it's called in asift.py
# (e.g., if it expects kp1, kp2, kp_pairs as separate args)
from code.asift.find_obj import init_feature, filter_matches, explore_match


def mosaicking(img0, img1, counter, h_all, H_tp):
    print("adding new frame test")
    h_all = inv(h_all)
    points0 = np.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]], dtype=np.float32)

    points1 = points1.reshape((-1, 1, 2))

    # get the transformed corner from new image
    points2 = cv2.perspectiveTransform(points0, h_all)

    print(points2)
    # get the max and min coordinate of mosaic images
    points = np.concatenate((points1, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)


    # additional translation from offset
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    #for homography in range(h_all):

    output_img = np.zeros(( y_max - y_min,x_max - x_min,3)) #define new global canvas
    output_img[-y_min:img1.shape[0] - y_min, -x_min:img1.shape[1] - x_min] = img1     # put old image in the bottom part

    warped_img = cv2.warpPerspective(img0, H_translation.dot(h_all),(x_max - x_min, y_max - y_min)) #apply homography to new image
    mask2 = (warped_img>0)*255
    mask3 = cv2.erode(mask2.astype('uint8'), np.ones((10,10), np.uint8))
    #mask3 = cv2.erode(mask2, numpy.ones((10,10), numpy.uint8))


    masked_mosaic = cv2.bitwise_and(np.uint8(output_img),  cv2.bitwise_not(np.uint8(mask3)))

    warped_img2 = cv2.bitwise_and(np.uint8(warped_img), np.uint8(mask3))



    output_img = cv2.bitwise_or(np.uint8(warped_img2),  np.uint8(masked_mosaic))


    return output_img, H_translation




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-hm', '--homography', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-save_path', dest='save_path', default="global_mosaic/", type=str, help="path to save result")
    # No -h or -d arguments are parsed here, as per maizaic_run.sh's new call
    args = parser.parse_args()

    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    result = None
    result_gry = None


    image_paths_raw = args.image_path # Keep raw args.image_path (might be a list or single dir)
    actual_image_files = [] # This list will hold paths to individual image files

    # Populate actual_image_files from input_image_paths
    for single_path_arg in image_paths_raw:
        if not os.path.exists(single_path_arg):
            print('Error: {0} does not exists!'.format(single_path_arg))
            continue
        if os.path.isdir(single_path_arg):
            extensions = [".jpeg", ".jpg", ".png"]
            for file_name in sorted(os.listdir(single_path_arg)):
                if os.path.splitext(file_name)[1].lower() in extensions:
                    actual_image_files.append(os.path.join(single_path_arg, file_name))
        else: # If it's a direct file path
            actual_image_files.append(single_path_arg)

    # Check if any images were found
    if not actual_image_files:
        print("Error: No images found in the specified -image_path(s). Cannot create global mosaic.")
        exit(1) # Exit if no images to process

    homography = args.homography # This argument is parsed but its associated code is commented out below
    image_index = -1
    counter = 0
    H = []
    points_in = np.array([[0,0], [0,0],[0,0],[0,0]], dtype=np.float32)

    '''
    # The code below to read homography from file is commented out.
    # If you intend to use pre-calculated homographies here, uncomment and ensure 'homography' variable
    # passed to my_asift is the correct global homography, not a per-frame one.
    # The current logic will calculate H using my_asift for each pair.
    with open(homography, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter = ",")
        for row in reader:
            H_each = np.asarray(row, dtype=np.float).reshape(3,3)
            H.append(H_each)

    print(H)
    '''
    H_tp = np.array([[0,0,0],[0,0,0],[0,0,0]])


    for image_path in actual_image_files: # Loop through the list of actual image file paths
        print("reading frame {0}".format(image_path))
        image_color_big = cv2.imread(image_path)

        if image_color_big is None:
            print(f"Error: Could not read image at {image_path}. Skipping.")
            continue # Skip to next image if imread fails

        filename = os.path.basename(image_path)
        height, width, channel = image_color_big.shape
        sw = int(width)
        sh = int(height)

        image_color = cv2.resize(image_color_big, (sw,sh))

        print(filename)

        image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

        image_index += 1

        if image_index == 0:
            print("inside image index", image_index)
            result = image_color
            prev_color = image_color # This variable seems unused after first frame.
            result_gry = image_gray
            continue

        print("counter ", counter)

        # image_color is new image
        # result is current global mosaic
        with Timer('my_asift'): # Using the Timer from common.py
            H = my_asift(result_gry, image_gray) # H is calculated using ASIFT between current mosaic and new image

        if H is None: # my_asift returns None if no homography found
            print(f"Warning: No homography found for {filename}. Skipping this frame.")
            continue

        H_flat = np.array(H).flatten().astype(np.float64)
        print(H_flat)
        print(args.save_path)

        # Saving homography for each pair
        with open(os.path.join(save_path, "H_asift.csv"), 'a', newline='') as f1: # Added newline='' for proper CSV writing
            wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
            wr.writerow(H_flat)

        # The line below was removed as 'elapsed_time_h' is not defined in this scope.
        # with open(os.path.join(save_path, "H_asift_time_elapsed.csv"), 'a', newline='') as f2:
        #     twr = csv.writer(f2, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
        #     twr.writerow([elapsed_time_h])

        result, H_tp = mosaicking(image_color, result, counter, H, H_tp) # img0 is new image, img1 is current mosaic

        counter+=1
        cv2.imwrite(os.path.join(save_path, f"global_mosaic_{counter}.png"), result) # Using f-string for clearer naming
        result_gry = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        prev_color = image_color # Still seems unused, but not harmful.

    # Final save after the loop finishes
    if result is not None: # Only save if at least one image was processed
        cv2.imwrite(os.path.join(save_path, f"final_global_mosaic_{counter}.png"), result)
    else:
        print("No final mosaic generated as no valid images were processed.")

    print("DONE!")
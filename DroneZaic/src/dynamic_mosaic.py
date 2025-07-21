#!/usr/bin/env python

# Built-in Modules
import os
import argparse
import logging
from datetime import datetime
import cv2
import csv
import image_stitching
from numpy import genfromtxt
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
    parser.add_argument('-start', '--start', dest='start', default = 0, type=int, help="stop stitching at")
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
    image_index = -1
    counter = 0
    
    H_cum = [] 
    H=[]
    cor2 = []
    corners_h = [] 
    temp_c_normalized = np.zeros((3,4))
    
    with open(homography_matrix, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter = ",")

        for row in reader:
            H_each = np.asarray(row, dtype=np.float).reshape(3,3)

            H.append(H_each)

            if counter == 0:
               H_temp = inv(H[counter])
               H_cum.append((H_temp))
               
            elif counter > 0:
               H_temp = np.dot((H_cum[counter-1]), inv(H[counter]))
               H_cum.append(H_temp)
            
            if counter == stop:
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
    for image_path in image_paths_list:
        image_rgb = cv2.imread(image_path)
        if image_rgb is None:
            #logging.error(f"Error reading image {image_path}")
            print("no image found")
            continue
        h, w = image_rgb.shape[:2]
        print(h, w)

        corners_4 = np.array([[0,0], [w,0],[w,h],[0,h]], dtype=np.float32)
        if count == 0:
            corners_h.append(corners_4.reshape((-1,1,2)))
            count += 1
        else:
            corners_h.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), H_cum[count-1]))
            count+=1

    corners_h_arr = np.asarray(corners_h)

    max_x = max(max_x, np.max(corners_h_arr[...,0].flatten()))
    min_x = min(min_x, np.min(corners_h_arr[...,0].flatten()))
    max_y = max(max_y, np.max(corners_h_arr[...,1].flatten()))
    min_y = min(min_y, np.min(corners_h_arr[...,1].flatten()))
    print(max_x, min_x, max_y, min_y)
    if min_x <= 0:
       offset_x = np.ceil(-(min_x))
       max_x += -min_x
    else:
       offset_x = 0#np.ceil(min_x)
       #max_x += min_x

    if min_y <= 0:
       offset_y = np.ceil(-(min_y))
       max_y += -min_y
    else:
       offset_y = 0#np.ceil(min_y)
       #max_y += min_y
    print(max_x, min_x, max_y, min_y)

    offset_matrix = np.matrix(np.identity(3), np.float32)
    offset_matrix[0,2] = offset_x
    offset_matrix[1,2] = offset_y
    print('offset matrix', offset_matrix)
    global_mosaic = np.zeros((int(np.floor(max_x)),int(np.floor(max_y)), 3), np.uint8)
    print('global mosaic size: ', np.shape(global_mosaic))
    
    row,col,channel = global_mosaic.shape
    mask = np.ones((col,row), np.uint8) 
    mask = mask*255

    for image_path in image_paths_list:
        image_rgb = cv2.imread(image_path)
        if image_rgb is None:
            #logging.error(f"Error reading image {image_path}")
            print("error no image")
            continue

        h, w = image_rgb.shape[:2]
        corners_4 = np.array([[1,1], [w,1],[w,h],[1,h]], dtype=np.float32)

        image_rgb = cv2.resize(image_rgb, (w,h))
        print(image_path)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
 
        print(image_index)
        
        if image_index == -1:
            print("this is executed")
            global_mosaic = cv2.warpPerspective(image_rgb, offset_matrix, (row, col))
            cor2.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), offset_matrix))
            
        elif image_index > -1:
            wrapped = cv2.warpPerspective(image_rgb, np.dot(offset_matrix, H_cum_new[image_index]), (row, col))
    
            (ret,data_map) = cv2.threshold(cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
            data_map = cv2.erode(data_map, np.ones((10,10), np.uint8))
            temp = cv2.add(global_mosaic, 0, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
            wrapped = cv2.add(wrapped, 0, mask=data_map, dtype=cv2.CV_8U)
            global_mosaic = cv2.add(temp, wrapped, dtype=cv2.CV_8U)
            cor2.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), np.dot(offset_matrix, H_cum_new[image_index])))
        
        image_index += 1
        
        image_stitching.helpers.display('mosaic_global2', global_mosaic)
        cv2.waitKey(200)

    print(save_path)
    cv2.imwrite(os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"_" + args.fname + ".png"), global_mosaic)
    end = time.time()

    print("time elapsed: "+  str(end-start)+  " seconds")


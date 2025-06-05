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
from asift import my_asift
from datetime import datetime
import time
import csv


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

    
    result = None
    result_gry = None

    image_paths = args.image_path
    pathss = args.image_path
    homography = args.homography
    image_index = -1
    counter = 0
    #H_each = np.array((3,3)) 
    H = [] 
    points_in = np.array([[0,0], [0,0],[0,0],[0,0]], dtype=np.float32)
    #points_in = points_in.reshape((-1, 1, 2))
    
    '''
    with open(homography, 'r') as csvFile:

        reader = csv.reader(csvFile, delimiter = ",")

        for row in reader:
            H_each = np.asarray(row, dtype=np.float).reshape(3,3)
            H.append(H_each)   
    

    print(H)
    '''
    H_tp = np.array([[0,0,0],[0,0,0],[0,0,0]])
    

    for image_path in image_paths:
        if not os.path.exists(image_path):
            logging.error('{0} is not a valid path'.format(image_path))
            continue
        if os.path.isdir(image_path):
            extensions = [".jpeg", ".jpg", ".png"]
            for file_path in sorted(os.listdir(image_path)):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    image_paths.append(os.path.join(image_path, file_path))
            continue

        logging.info("reading image from {0}".format(image_path))
        image_color_big = cv2.imread(image_path)
        filename = os.path.basename(image_path)
        height, width, channel = image_color_big.shape
        sw = int(width/args.scale)
        sh = int(height/args.scale)

        image_color = cv2.resize(image_color_big, (sw,sh))

        print(filename)

        cv2.imwrite(os.path.join(frames_to_mosaic_path,filename), image_color)
        
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
        
        image_index += 1

        if image_index == 0:
            prev_gray = image_gray
            #features0 = sift.detectAndCompute(result_gry, None)
            continue

        print("counter ", counter)

        #image_color is new image
        #result is global mosaic

        h_time = time.time()
        H = my_asift(prev_gray, image_gray)
        elapsed_time_h = time.time()-h_time

        H_flat = np.array(H).flatten().astype(np.float64)
        print(H_flat)
        print(args.save_path)        
        with open(hm_path +"/H_asift.csv", 'a') as f1:
           wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
           wr.writerow(H_flat)

        with open(hm_path+"/H_asift_time_elapsed.csv", 'a') as f2:
           twr = csv.writer(f2,  delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
           twr.writerow([elapsed_time_h])
         

        '''   
        H_curr = np.asarray(H[counter])
        print(H_curr)
        if counter==0:
            H_acum = H_curr
        else:
            H_acum = np.dot((H_acum), (H_curr))
            #H_acum = np.dot((H_curr), H_acum)i

        '''
        #result, H_tp = image_stitching.combine_images(image_color, result, counter, H, H_tp)

        if args.display and not args.quiet:
            image_stitching.helpers.display('result', result)
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break
        counter+=1
        
        #image_stitching.helpers.save_image(args.save_path+"Frame_"+str(counter)+".png", result)
  
        prev_gray = image_gray
    logger.info("processing complete!")
        
    if args.display and not args.quiet:
        cv2.destroyAllWindows()
    if args.save:
        logger.info("saving stitched image to {0}".format(args.save_path))
        image_stitching.helpers.save_image(args.save_path, result)

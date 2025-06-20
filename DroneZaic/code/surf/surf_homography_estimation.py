'''
author: Dewi Kharismawati

this experiment of mosaicking with surf:
    - estimating homograhy matrices
    - then saving to the csv file in desired directory
'''
import os
import argparse
from datetime import datetime
import cv2
import csv
from numpy import genfromtxt
import numpy as np
from numpy.linalg import inv
from surf import surf
from datetime import datetime
import time
import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-hm', '--homography', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-save_path', dest='save_path', default="homography_matrices/", type=str, help="path to save result")
    parser.add_argument('-scale', '--scale', dest='scale', default=1, type=int, help="size down scale ratio")
    args = parser.parse_args()

    
    result = None
    result_gry = None

    frames_to_mosaic_path = args.save_path + '/frames_to_mosaic/'
    hm_path = args.save_path + '/homography_matrices/'

    if not os.path.exists(frames_to_mosaic_path):
         os.makedirs(frames_to_mosaic_path)

    if not os.path.exists(hm_path):
         os.makedirs(hm_path)



    image_paths = args.image_path
    pathss = args.image_path
    homography = args.homography
    scale = args.scale
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
            print('{0} does not exist'.format(image_path))
            continue
        if os.path.isdir(image_path):
            extensions = [".jpeg", ".jpg", ".png"]
            for file_path in sorted(os.listdir(image_path)):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    image_paths.append(os.path.join(image_path, file_path))
            continue

        print("reading current frame from {0}".format(image_path))
        image_color_big = cv2.imread(image_path)
        filename = os.path.basename(image_path)
        height, width, channel = image_color_big.shape
        sw = int(width/scale)
        sh = int(height/scale)

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
        H = surf(prev_gray, image_gray)
        elapsed_time_h = time.time()-h_time

        H_flat = np.array(H).flatten().astype(np.float64)
        print(H_flat)
        print(args.save_path)        
        with open(hm_path+"/H_surf.csv", 'a') as f1:
           wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
           wr.writerow(H_flat)

        with open(hm_path+"/H_surf_time_elapsed.csv", 'a') as f2:
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
  
        counter+=1
        
        prev_gray = image_gray
    print("homography matrices has been saved to " + hm_path)

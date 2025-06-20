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
from asift import my_asift
from datetime import datetime
import time
import csv



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
    output_img[-y_min:img1.shape[0] - y_min, -x_min:img1.shape[1] - x_min] = img1    # put old image in the bottom part

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
    args = parser.parse_args()

    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
            print('{0} does not exists!'.format(image_path))
            continue
        if os.path.isdir(image_path):
            extensions = [".jpeg", ".jpg", ".png"]
            for file_path in sorted(os.listdir(image_path)):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    image_paths.append(os.path.join(image_path, file_path))
            continue

        print("reading frame {0}".format(image_path))
        image_color_big = cv2.imread(image_path)
        filename = os.path.basename(image_path)
        height, width, channel = image_color_big.shape
        sw = int(width)
        sh = int(height)

        image_color = cv2.resize(image_color_big, (sw,sh))

        print(filename)

        #cv2.imwrite(os.path.join(args.save_path,filename), image_color)
        
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
        
        image_index += 1

        if image_index == 0:
            print("inside image index", image_index)
            result = image_color
            prev_color = image_color
            result_gry = image_gray
            continue

        print("counter ", counter)

        #image_color is new image
        #result is global mosaic
         
        h_time = time.time()
        H = my_asift(result_gry, image_gray)
        elapsed_time_h = time.time()-h_time

        H_flat = np.array(H).flatten().astype(np.float64)
        print(H_flat)
        print(args.save_path)        
        with open(save_path+"/H_asift.csv", 'a') as f1:
           wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
           wr.writerow(H_flat)

        with open(save_path+"/H_asift_time_elapsed.csv", 'a') as f2:
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
        
        result, H_tp = mosaicking(image_color, result, counter, H, H_tp)

        counter+=1
        cv2.imwrite(save_path+"/global_mosaic_"+str(counter)+".png", result)
        result_gry = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        prev_color = image_color

    cv2.imwrite(save_path+"/final_global_mosaic_"+str(counter)+".png", result)
    
    print("DONE!")
       

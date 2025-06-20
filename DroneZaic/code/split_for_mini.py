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
import shutil
from scipy.signal import find_peaks, medfilt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


def move_images(image_path, save_path, homography, boundaries, duplicate, overlap):
    image_files = []
    prev_images = []
    H = []
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image_path in image_path:
       if os.path.isdir(image_path):
         extensions = [".jpeg", ".jpg", ".png", ".tif"]
         for file_path in sorted(os.listdir(image_path)):
            if os.path.splitext(file_path)[1].lower() in extensions:
                image_files.append(os.path.join(image_path, file_path))

                
    with open(homography, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter = ",")
        for row in reader:
            H_each = np.array(row).astype(np.float64)
            flattened_list = H_each.ravel().tolist()
            #print(flattened_list)
            H.append(flattened_list)

    #H = np.asarray(H)

    #print(H)
            
    group_boundary = boundaries
    print('group boundary', group_boundary)
    group_count = 0
    subfolder_count = 1
    boundary = 0
    subfolder = os.path.join(save_path, 'group_{}'.format(str(subfolder_count).zfill(3)))
    print(subfolder)
    i = 0
    start = 0
    
    while i < len(image_files):
         #print('start: ', start)
         #print(i)
         filename = os.path.basename(image_files[i])
         #print(filename)

         if group_count == 0 and not os.path.exists(subfolder):
            os.makedirs(subfolder)

                
         destination_path = os.path.join(subfolder, filename)
         #print(destination_path)
         #print(duplicate)
         if not duplicate:
             #print("test move")
             shutil.move(image_files[i], destination_path)					 
         else:
             #print("test copy")
             shutil.copy(image_files[i], destination_path)
         group_count += 1
         i = i+1


         
         if i == group_boundary[boundary]:
            h_temp = H[start:i-1]
            boundary += 1



            with open(save_path+"/H_asift_"+'group_{}'.format(str(subfolder_count).zfill(3))+".csv", 'a') as f1:
               wr = csv.writer(f1, delimiter=",", quoting = csv.QUOTE_NONE) # escapechar = " ",
               for h_each_save in h_temp:
                   wr.writerow(h_each_save)

            i = i-overlap
            #print(i)
            group_count = 0
            subfolder_count += 1
            subfolder = os.path.join(save_path, 'group_{}'.format(str(subfolder_count).zfill(3)))

            start = i
            h_temp = []

    '''        
    h_temp = H[start:i]
    with open(save_path+"/H_asift_"+'group_{}'.format(subfolder_count)+".csv", 'a') as f1:
        wr = csv.writer(f1, delimiter=",", escapechar = ",", quoting = csv.QUOTE_NONE)
        wr.writerow(h_temp)
    '''

def filter_on_shoulder(peaks, properties, data, window_size):
    filtered_peaks = []
    for peak in peaks:
        
        start = max(0, peak - window_size)
        end = min(len(data), peak + window_size)
        print(peak, start, end)        
        if data[peak] == max(data[start:end]):
            filtered_peaks.append(peak)
    return filtered_peaks


def median_filter_smoothing(data, window_size):

    return medfilt(data, window_size)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


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
   plt.plot(filtered_peaks, angle_diffs[filtered_peaks], "x", label='peaks', color = 'red')
   plt.title("data with detected peaks")
   plt.xlabel("frame number")
   plt.ylabel("angle difference")
   plt.legend()
   plt.savefig(save_path)
   #plt.show()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-save_path', dest='save_path', default="RESULTS/global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
    parser.add_argument('-hm', '--homography', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-angle_csv', '--angle_csv', type=str, nargs='+', help='csv file that stores the angle difference csv')
    parser.add_argument('-duplicate', dest='duplicate', action='store_true', help='Enable duplication. Default is disable.')
    args = parser.parse_args()

    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    partition_boundary = []

    #if we have more than 1 videos
    i = 0
    prev_end_number = 0

    
    
    for angle_file in args.angle_csv:
        angle_diffs = []
        
        with open(angle_file, 'r') as file:
          i+=1
          angle_reader = csv.reader(file)
          for row in angle_reader:
             angle_diffs.append(float(row[0]))

        #evaluate the angle diff for boundaries per file
        angle_diffs = np.asarray(angle_diffs)
       
        #window_size = 5 

        thresh = find_threshold(angle_diffs)

        filtered_peaks, properties = find_peaks(angle_diffs, thresh, distance=5)

        #filtered_peaks = filter_on_shoulder(peaks, properties, angle_diffs, window_size)
        
        plot_partition(angle_diffs, filtered_peaks, os.path.join(save_path, 'angle_peaks_plot%02d.png' % i)) 
        
        print(filtered_peaks)
        print(len(angle_diffs)-1)
        print(angle_diffs)
        
        #filtered_peaks(len(angle_diffs)-1)


        #convert to array to allow addition
        current_peaks = np.asarray(filtered_peaks)+3+prev_end_number

        #convert back to list

        partition_boundary += current_peaks.tolist()
        
        print("Frame boundaries so far : ", partition_boundary)
        
        prev_end_number += len(angle_diffs)+2
        print('prev end number: ', prev_end_number)
        
    


    partition_boundary.append(prev_end_number)
    print(partition_boundary)
    
    #this is for partition 12.6 24 passes
    #partition_boundary=[110,193,256,355,448,583,661,758,767,850,981,1072,1226,1315,1424,1499,1600]
    

    #this is for partition of 12.6 15 passes manual split
    #partition_boundary=[99,184,292,376,475,573,691,770]    
    #partition_boundary=[288,320,756]
    #partition_boundary = [133,143,283]
    #grouping the group partition
    move_images(args.image_path, save_path, args.homography, partition_boundary, args.duplicate, overlap=0)
        


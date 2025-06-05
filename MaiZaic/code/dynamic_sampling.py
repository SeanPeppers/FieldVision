'''

author: Dewi Kharismawati
project: MaiZaic


call this by:

- for video
python dynamic_sampling.py -video /path/to/raw/video -save_path /path/to/where/you/want/to/save/the/quiver/plot



- for frames
python dynamic_sampling.py -image_path /path/to/raw/images -save_path /path/to/where/you/want/to/save/the/quiver/plot


'''


import cv2
import argparse
from datetime import datetime
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import time



def detect_cam_movement_video(video, save_path, scale, i, fps, win_size, ss):

    translation_threshold = 5
    quiver_path = os.path.join(save_path, 'quiver')
    distribution_path = os.path.join(save_path, 'distribution')
    raw_path =  os.path.join(save_path, 'raw')

    
    if not os.path.exists(quiver_path):
        os.makedirs(quiver_path)

    if not os.path.exists(distribution_path):
        os.makedirs(distribution_path)


    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        
    video = cv2.VideoCapture(video)

    current_fps = fps


    
    default_interval = int(video.get(cv2.CAP_PROP_FPS) * current_fps)
    current_interval = default_interval
    if ss==0:
       frame_index = default_interval
    else:
       frame_index = ss*30

    ret = video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)


    #read the first frame
    ret, init_frame = video.read()

    #i = 0
    
    cv2.imwrite(os.path.join(raw_path, args.fname+'_frame_%06d.png' % i), init_frame)
    #init_frame = init_frame.astype('uint8')
    prev_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape



    new_height = int(np.round(height/scale))
    new_width = int(np.round(width/scale))


    prev_gray = cv2.resize(prev_gray, (new_width, new_height))

    j = 0
    non_translation_index = 0
    degree_mean = []


    translation = True

    while True:

        print("================================")

        print('frame index', frame_index)

        print('skip index counter j : ', j)



        ret = video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
 
        ret, frame = video.read()

        if not ret:
            print('end of video')
            break

        frame_time_sec = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        print('frame time in seconds right now: ', frame_time_sec)


        gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray_original, (new_width, new_height))

        #print(gray.shape)

        #calculate dense optical flow using farneback
        t = time.time() 
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 150, win_size, 3, 5, 1.2, 0)
        elapsed_flow_time = time.time()-t

        print('********elapsed time for optical flow: ', elapsed_flow_time)

        print('fps: ', fps)
        print('win_size: ', win_size)
        #height, width = flow.shape[:2]
        y, x = np.mgrid[0:new_height, 0:new_width]

        #extract motion vectors for x and y vector
        flow_x = flow[y, x, 0]
        flow_y = flow[y, x, 1]

        overlap = compute_overlap(np.abs(flow_x), np.abs(flow_y))

        #this statement for rotation extract everything
        '''
        if non_translation_index == 0:
            translation_threshold = 5
        else:
            translation_threshold = 0.1
        '''

        #detect camera movement with detecting translation, and modify extraction interval
        translation = evaluate_trajectory(np.abs(flow_x), np.abs(flow_y), translation_threshold)


        print('translation is ', translation)

        ''' 
        okay, this non_translation_index==1 is implemented in the if statement to avoid the infinity loop back and forth.
        
        to avoid this scenario:  
              - base_frame (say f0) vs current_frame (say f14) is non translation is detected, so interval is back to half of current interval.
              - base_frame (f0) vs current_frame (f7), if f0 and f7 is translation and the overlap is > 98, it will go forward. This will go to infinity loop
               
        so my solution:
              - base_frame (say f0) vs current_frame (say f14) is non translation is detected, so interval is back to half of current interval.
              - pushing to save f7, then, keep extracting with half of the default interval until we have translation again

        '''


        if translation == False or non_translation_index == 1:
           print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

           if non_translation_index == 0:
                current_interval = np.floor(current_interval/2)
                frame_index = frame_index - current_interval+1
                #current_interval = 1#np.floor(current_interval/2)  
                non_translation_index += 1                

                print('it is not translation, moving the index back at %d current interval is %d' % (frame_index, current_interval))

                continue
           
           frame_index = frame_index + current_interval

           print('it is not translation, current interval is %d' %  current_interval)


           i += 1
           j = 0
           print('saving frames to : ', os.path.join(raw_path, args.fname+'_frame_%06d.png' % i))

           cv2.imwrite(os.path.join(raw_path, args.fname+'_frame_%06d.png' % i), frame)
           prev_gray = gray.copy()

                
           mean_direction_in_degree = compute_direction_save_plots(flow_x, flow_y, flow, i, quiver_path, distribution_path, args.fname) 
            
           degree_mean.append(mean_direction_in_degree)               
              
           non_translation_index += 1

           continue
       
        elif translation == True:
           print('default interval: ', default_interval)
           current_interval = default_interval
           non_translation_index = 0

        #overlap decision
        if 0.85 <= overlap <= 0.98:
            i += 1

            print('this frame has %f overlap' % (overlap*100))
            

            #saving_frame
            print('saving the  frame ', os.path.join(raw_path, args.fname+ '_frame_%06d.png' % i))
            cv2.imwrite(os.path.join(raw_path, args.fname+'_frame_%06d.png' % i), frame)

            frame_index += default_interval

            prev_gray = gray.copy()

            mean_direction_in_degree = compute_direction_save_plots(flow_x, flow_y, flow, i, quiver_path, distribution_path, args.fname) 
            
            degree_mean.append(mean_direction_in_degree)

            j = 0



        elif 0.98 < overlap <= 1:
            #reduce the sampling rate
            print('overlap is over:  %f' % (overlap*100))
            print('reducing the sampling rate')
            
            frame_index = frame_index + np.floor(default_interval/2)
           
            
            
            if j >= 4:
                j = 0
                i += 1

                print('saving frames to : ', os.path.join(raw_path, args.fname+'_frame_%06d.png' % i))

                cv2.imwrite(os.path.join(raw_path, args.fname+'_frame_%06d.png' % i), frame)
                prev_gray = gray.copy()

                
                mean_direction_in_degree = compute_direction_save_plots(flow_x, flow_y, flow, i, quiver_path, distribution_path, args.fname) 
            
                degree_mean.append(mean_direction_in_degree)

                        
                continue

            j += 1


        else:
            #increase the sampling rate 
            print('overlap is lower:  %f' % (overlap*100))
            print('current time increasing the sampling rate')
            #go back to half second
            current_interval = np.ceil(current_interval/2)
            if (current_interval) > 1:
                frame_index = frame_index - current_interval
            else:
                print('uh oh, current frame is the adjecent of the previous frame')
                print('saving frames to : ', os.path.join(raw_path, args.fname+'_frame_%06d.png' % i))

                cv2.imwrite(os.path.join(raw_path, args.fname+'_frame_%06d.png' % i), frame)
                prev_gray = gray.copy()

                mean_direction_in_degree = compute_direction_save_plots(flow_x, flow_y, flow,i, quiver_path, distribution_path, args.fname) 
            
                degree_mean.append(mean_direction_in_degree)
                j = 0
                


    degree_diff = compute_degree_diff(np.array(degree_mean))
    print("******")
    print(degree_diff)
    degree_plot(np.array(degree_diff), os.path.join(save_path, "plot_of_diff_angle.png"))
    


    with open(args.save_path + '/'+args.fname+'_angle_diff.csv', 'a') as f1:

        for angle_difff in degree_diff:

             #print(angle_difff)
             wr = csv.writer(f1, quoting = csv.QUOTE_NONE)
             wr.writerow([angle_difff])



    #compute threshold to get the first mini mosaic frame
    #thresh, degree_diff_thresholded = cv2.threshold(np.asarray(degree_diff), 0, 360, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #print("threhold value: ", thresh)


def compute_direction_save_plots(flow_x, flow_y, flow, index, quiver_path, distribution_path, fname):
    
    #Compute magnitude and direction of motion vectors
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    mean_magnitude = compute_iqr_average(magnitude)
    #print('mean magnitude ', mean_magnitude

    direction = np.arctan2(flow_y, flow_x)
    direction_degree = ((np.degrees(direction)+360)%360)
    mean_directions = compute_iqr_average(direction)
    mean_direction_in_degree = ((np.degrees(mean_directions)+360)%360)
    #print('mean_direction', mean_degree)


    #drawing quiver plot and save it in the save image path
    draw_quiver(flow, magnitude, 50, os.path.join(quiver_path, fname+ '_quiver_frame_%06d.png' % index))


    #uncomment for plotting distribution
    pixel_distribution_plot(magnitude, direction_degree, os.path.join(distribution_path, fname+ '_distribution_frame_%06d.png' % index))


    return mean_direction_in_degree
     


def draw_quiver(flow, magnitude, skip, save_path):


    filename = int(save_path[-10:-4])
    prev_filename = int(filename)-1
    #print(filename, prev_filename)
    
    height, width = flow.shape[:2]
    y, x = np.mgrid[0:height:skip, 0:width:skip]
    
    min_color = 0
    max_color = 100


    tick_positions = np.linspace(min_color, max_color, max_color/10+1)

    #extract motion vectors at the specified grid points
    #flow_x = flow[y, x, 0]
    #flow_y = flow[y, x, 1]

    #compute magnitude and direction of motion vectors
    #magnitude = np.sqrt(flow_x**2 + flow_y**2)
    #direction = np.arctan2(flow_y, flow_x)

    #print(direction[10,:])

    #create quiver plot
    plt.figure(figsize=(10, 8))
    plt.quiver(x, y, flow[y, x, 0], flow[y, x, 1], magnitude, pivot='mid', cmap=plt.cm.jet, linewidth=5, headwidth=3)
    plt.colorbar(label='Magnitude', ticks=tick_positions, format='%.2f', orientation='vertical', shrink=0.57)
    plt.clim(min_color, max_color)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Quiver from Frame %.3d to %.3d (plot every %dth vector)' % (prev_filename, filename, skip))

    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    #plt.show()

    
    plt.savefig(save_path)
    plt.close()


def compute_iqr_average(data):
   
    #okay we are using radians because the data is circular 360 and 1 are close to each other
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    #print(q1, q3)


    data_iqr = np.mean(data[np.logical_and(data>=q1,data<=q3)])
    #print(data_iqr)


    return data_iqr

def evaluate_trajectory(x, y, thresh):

    magnitude = np.sqrt(x**2 + y**2)
    #print(magnitude)
    height, width = magnitude.shape

    #top_left = (2*int(height/6), 2*int(width/6))
    #bottom_right = (height-2*int(height/6), width-2*int(width/6))

    #inner rectangle
    inner = magnitude[2*int(height/6):height-2*int(height/6), 2*int(width/6):width-2*int(width/6)]

    avg_inner = np.mean(inner, axis=(0,1))

    #outer per area
    top_rec = np.mean(magnitude[:int(height/6),:], axis=(0,1))
    bottom_rec = np.mean(magnitude[height-int(height/6):, :], axis=(0,1))
    left_rec = np.mean(magnitude[int(height/6):height-int(height/6), :int(width/6)], axis=(0,1))
    right_rec = np.mean(magnitude[int(height/6):height-int(height/6), width-int(width/6):], axis=(0,1))
    

    avg_outer = (top_rec+bottom_rec+left_rec+right_rec)/4

    #print(avg_inner, avg_outer)
    print("diff ", np.abs(avg_inner-avg_outer))
    if np.abs(avg_inner-avg_outer)<thresh:
        translation = True
    else:
        translation = False

        

    '''
    #outer rectangle
    #outer = np.concatenate([magnitude[:int(height/6),:], \
                             magnitude[height-int(height/6), :], \ 
                             magnitude[int(height/6):height-int(height/6), :int(width/6)], \
                             magnitude[int(height/6):height-int(height/6), width-int(width/6):]])
    '''
    #print(magnitude[:int(height/6),:])
    return translation

def compute_overlap(x, y):
    height, width = x.shape
    #print(x.shape)
    displacement_x = compute_iqr_average(x)
    displacement_y = compute_iqr_average(y)

    overlap_width = width-displacement_x
    overlap_height = height-displacement_y
    
    #print('***********')
    #print('overlap width', displacement_x)
    #print('overlap height', displacement_y)

    overlap_percentage = (overlap_width*overlap_height)/(width*height)
    '''
    if 0.8 <= overlap_percentage <= 0.95:
        print('this frame has %f overlap' % (overlap_percentage*100))
        print('saving the frame')
    else:
        #go back to previous frame
        print('statement in else')
        print('overlap is only %f' % (overlap_percentage*100))
    '''
 
    return overlap_percentage 



def compute_degree_diff(mean_degree1):

    mean_degree = np.asarray(mean_degree1)
    degree_diff_array = []

    
    for i in range(1, len(mean_degree)):

       #min_direction_diff = min((360.0 - max(mean_degree[i-1], mean_degree[i]) + min(mean_degree[i-1], mean_degree[i])), (np.abs(mean_degree[i-1] - mean_degree[i])).astype(float))

       diff = abs(mean_degree[i] - mean_degree[i-1])
       if diff > 180:
           min_direction_diff = 360 - diff
       else:
           min_direction_diff = diff
       
       degree_diff_array.append(min_direction_diff)


    return degree_diff_array



def pixel_distribution_plot(magnitude, direction, save_path):
    #print('pixel_dist')

    mag = magnitude.flatten()
    dirctn = direction.flatten()
    #print(np.min(dirctn), np.max(dirctn))

    #print(mag)
    #print("=========")
    #print(dirctn)


    plt.figure(figsize=(10,8))
    
    plt.hist(mag, bins=100, density = False, color='blue', alpha=0.5, label='Magnitude')


    plt.hist(dirctn, bins = 100, density = False, color='red', alpha =0.5, label='Direction')


    plt.xlim(0,360)
    plt.ylim(0, mag.shape[0])
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Distribution of Magnitude and Direction')

    #plt.show()

    plt.savefig(save_path)
    plt.close()


def compute_mad(data):

    median = np.median(data)

    absolute_deviation = np.abs(data-median)

    mad = np.median(absolute_deviation)

    return mad

def degree_plot(degrees, save_path):

    x = np.arange(len(degrees))

    plt.figure(figsize=(10, 10))

    plt.plot(x, degrees)


    plt.xlabel('frame no')
    plt.ylabel('degree')
    plt.title('plot of the degree angle between frames')

    plt.savefig(save_path)
    

    plt.show()
    #plt.close()


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('-image_path', type=str, help="paths to one or more images or image directories")
    parser.add_argument('-video', type=str, help="paths to one video")
    parser.add_argument('-save_path', type=str,  dest='save_path', default="RESULTS/global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),  help="path to save result")
    parser.add_argument('-hm', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-scale', type=int, dest='scale', default=1, help='the downsampled scale for the frame')
    parser.add_argument('-fps', type=float, dest='fps', default=0.5, help='the downsampled scale for the frame')
    parser.add_argument('-win', type=int, dest='win_size', default=50, help='the downsampled scale for the frame')
    parser.add_argument('-start_number', type=int, dest='start_number', default=1, help='initial number to save the frame id')
    parser.add_argument('-ss', type=int, default=0, help='where do you want the start time to extract')
    parser.add_argument('-fname', type=str, help="desired prefix name for frame extracted")

    args = parser.parse_args()
    #print(args.image_path)

    #detect_camera_movement(args.image_path, args.save_path, args.scale)


    detect_cam_movement_video(args.video, args.save_path, args.scale, args.start_number, args.fps, args.win_size, args.ss)

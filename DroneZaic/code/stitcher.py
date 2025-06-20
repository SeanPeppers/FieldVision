#!/usr/bin/env python

import os
import argparse
from datetime import datetime
import cv2
import csv
from numpy import genfromtxt
import numpy as np
from numpy.linalg import inv
import time
import sys
import copy
import exifread
from math import radians, cos, sin, sqrt, atan2
from rasterio.transform import rowcol
import rasterio
from rasterio.transform import xy

np.set_printoptions(threshold=sys.maxsize)

def load_raster_transform(image_path):
    if image_path.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(image_path) as src:
                transform = src.transform
            return transform
        except Exception as e:
            print("Error loading transform for .tiff file: %s" % str(e))
            return None
    else:
        print("Skipping transform loading for non-tiff file.")
        return None


def gps_to_pixel(gps_lat, gps_lon, transform):
    row, col = rowcol(transform, gps_lon, gps_lat)
    return (col, row)  

def pixel_to_gps(pixel_x, pixel_y, transform):
    lon, lat = xy(transform, pixel_y, pixel_x)
    return lat, lon
	
def get_decimal_from_dms(dms, ref):
    degrees = float(dms[0].num) / dms[0].den
    minutes = float(dms[1].num) / dms[1].den
    seconds = float(dms[2].num) / dms[2].den
    dec = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        dec = -dec
    return dec

def extract_gps_from_image(image_path):
    gps = None
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                lat = get_decimal_from_dms(tags['GPS GPSLatitude'].values, str(tags['GPS GPSLatitudeRef']))
                lon = get_decimal_from_dms(tags['GPS GPSLongitude'].values, str(tags['GPS GPSLongitudeRef']))
                gps = (lat, lon)
    except Exception as e:
        print("Error extracting GPS: %s" % str(e))
    return gps

def gps_error(gps1, gps2):
    R = 6371000
    lat1, lon1 = radians(gps1[0]), radians(gps1[1])
    lat2, lon2 = radians(gps2[0]), radians(gps2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def apply_homography_to_gps(gps, H):
    x, y = gps
    pt = np.array([x, y, 1.0], dtype=np.float64)  
    pt_proj = np.dot(H, pt)
    pt_proj = np.asarray(pt_proj).flatten()  
    pt_proj /= pt_proj[2] 
    return (pt_proj[0], pt_proj[1])

def save_mosaic_with_gps(global_mosaic, save_path, args, gps_projected=None, gps_actual=None, final_frame_path=None):
    if gps_projected and gps_actual:
        error_m = gps_error(gps_projected, gps_actual)
        print("Projected GPS:", gps_projected)
        print("Actual GPS:", gps_actual)
        print("GPS Error (meters):", error_m)

        proj_x, proj_y = int(gps_projected[0]), int(gps_projected[1])
        cv2.circle(global_mosaic, (proj_x, proj_y), 10, (0, 255, 255), -1)

        if gps_actual:
            act_x, act_y = int(gps_actual[0]), int(gps_actual[1])
            cv2.rectangle(global_mosaic, (act_x-10, act_y-10), (act_x+10, act_y+10), (0, 0, 255), 3)

    out_path = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + args.fname + ".png")
    cv2.imwrite(out_path, global_mosaic)
    print("Saved mosaic to:", out_path)

    if final_frame_path and gps_projected:
        with rasterio.open(final_frame_path) as src:
            transform = src.transform
            lat, lon = pixel_to_gps(gps_projected[0], gps_projected[1], transform)
            print("Projected pixel coordinate corresponds to GPS:", (lat, lon))


    

	
def display_mosaic(fname, img):
    max_size = 300000
    scale = np.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow('test test', np.uint8(img))
    cv2.waitKey(100) 

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help='paths to one or more frames')
    parser.add_argument('-start', '--start', dest='start', default = 0, type=int, help="start stitching at")
    parser.add_argument('-stop', '--stop', dest='stop', default = 10000, type=int, help="stop stitching at")
    parser.add_argument('-save_path', dest='save_path', default="results/global_mosaic", type=str, help="path to save result")
    parser.add_argument('-r', '--rho', dest='r', default=10, type=int, help="directory value")
    parser.add_argument('-hm', '--homography', type=str, help='txt or csv file that stores homography matrices')
    parser.add_argument('-fname', '--fname', dest='fname', default='global_mosaic_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), help='desired filename for the global mosaic')
    parser.add_argument('-video', '--videos', dest = 'video', type=str, default= 'N', help='do you want to save frames addition process for videos?')
    parser.add_argument('-scale', '--scale', dest='scale', default=1, type=float, help='image size scale')
    parser.add_argument('-mini_mosaic', '--mini_mosaic', dest='mini_mosaic', action='store_true', help='enable mini mosaic')
    

    args = parser.parse_args()

    save_path = args.save_path
    print(args.mini_mosaic)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    if args.mini_mosaic:
        #mini_path = os.path.join(args.save_path, 'old_cornet_mini_mosaics')
        mini_path = args.save_path
        if not os.path.exists(mini_path):
             os.makedirs(mini_path)

        save_path = mini_path

    result = None
    result_gry = None
				
    '''
    #w = 3959
    #h = 2014
    w = 1354
    h = 710
    #w = 2048
    #h = 1080 
    #w = 1318
    #h = 680
     
    if args.r == 10 :
       w = 1327#1353
       h = 655#666
    elif args.r == 11 or args.r == 12:
       w = 1334
       h = 669
    else:
       w = 1353
       h = 666
    '''

    
    stop = args.stop
    homography_matrix = args.homography
    image_paths = args.image_path
    image_index = -1
    counter = 0


    all_img = sorted([f for f in os.listdir(image_paths[0]) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    initial_img_path = os.path.join(image_paths[0], all_img[0])
    img = cv2.imread(initial_img_path)
    h, w, _ = img.shape
    w = int(np.round(w/args.scale))
    h = int(np.round(h/args.scale))
    img = cv2.resize(img, (w,h))
    print('Image dimensions (h, w):', h, w)
		
    H_cum = [] 
    H=[]
    cor2 = []





    if initial_img_path.lower().endswith(('.tif', '.tiff')):
       transform = load_raster_transform(initial_img_path)
    else:
       transform = None  # No transform for .jpg images

    #transform = load_raster_transform(initial_img_path)
    gps_initial = extract_gps_from_image(initial_img_path)
    gps_projected = None
    gps_actual = None
    print(gps_initial)




    #pred_paths = []
    #pred_per_image = []
    #pred_square = []
    #pred_square_per_image = []
    
    
    corners_h = []
    corners_4 = np.array([[1,1], [w,1],[w,h],[1,h]], dtype=np.float32)
    #print(corners_4)
    temp_c_normalized = np.zeros((3,4))
    
    with open(homography_matrix, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter = ",")

        for row in reader:
            H_each = np.asarray(row, dtype=np.float).reshape(3,3)

            H.append(H_each)

            #print(H_each)


            if counter == 0:
               H_temp = inv(H[counter])
               #H_temp = H_temp/H_temp[2,2]
               H_cum.append(( H_temp))
               
            elif counter > 0 :
               
               H_temp = np.dot((H_cum[counter-1]), inv(H[counter]))
               #H_temp = H_temp/H_temp[2,2]
               H_cum.append(H_temp)
            
            if counter == stop:
               break
            
            #print(counter)
            #print(H_temp)

            #corner_temp = cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), (H_cum[counter]))
            #print(corner_temp)
            #corners_h.append(corner_temp/[corner_temp[2,:] ,corner_temp[2,:], corner_temp[2,:], corner_temp[2,:]]) 
            corners_h.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), (H_cum[counter])))            
        
            counter = counter+1
            #print('corners after warped:', corners_h[counter-1])
    H_cum_new = np.asarray(H_cum)
    corners_h_arr = np.asarray(corners_h)

    #x = width= col; y=hight=row
    max_x = np.max(corners_h_arr[...,0].flatten())
    min_x = np.min(corners_h_arr[...,0].flatten())

    max_y = np.max(corners_h_arr[...,1].flatten())
    min_y = np.min(corners_h_arr[...,1].flatten())

    #print(max_x, min_x, max_y, min_y)

    if min_x<=0:
       offset_x = np.ceil(-(min_x))
       max_x += -min_x
    else:
       offset_x = 0#np.ceil(min_x)
       #max_x += 0#min_x

    if min_y<=0:
       offset_y = np.ceil(-(min_y))
       max_y += -min_y
    else:
       offset_y = 0#np.ceil(min_y)
       #max_y += 0#min_y



    offset_matrix = np.matrix(np.identity(3), np.float32)
    offset_matrix[0,2] = offset_x
    offset_matrix[1,2] = offset_y
    print(offset_matrix)


    #print(np.ceil(max_x+offset_x))
    #define global mosaic
    global_mosaic = np.zeros((int(np.floor(max_x)),int(np.floor(max_y)), 3), np.uint8)
    print('global mosaic size: ', np.shape(global_mosaic))
    
    row,col,channel = global_mosaic.shape
    mask = np.ones((col,row), np.uint8) 
    mask = mask*255
    #save_mosaic = np.zeros((int(np.floor(max_y))+h,int(np.floor(max_x))+w, 3), np.uint8)
    nh = (int(np.floor(max_y))+h)
    nw = (int(np.floor(max_x))+w)
    if (nh % 2) != 0:
       nh = nh + 1


    if (nw % 2) != 0:
       nw = nw + 1

    save_mosaic = np.zeros((nh,nw, 3), np.uint8)

    r,c,ch = save_mosaic.shape
    pred = ".txt"

    #print(mask)
    #print(row, col)
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print('{0} path does not exists'.format(image_path))
            continue
        if os.path.isdir(image_path):
            extensions = [".jpeg", ".jpg", ".png", ".JPG", ".tif", ".tiff"]
            #extensions = [".png"]

            for file_path in sorted(os.listdir(image_path), reverse=False):
                if os.path.splitext(file_path)[1].lower() in extensions:
                    image_paths.append(os.path.join(image_path, file_path))
                    
                    
                #if os.path.splitext(file_path)[1].lower() in pred:
                #    pred_paths.append(os.path.join(image_path, file_path))
            continue


        
        
        '''
        #read the prediction txt file
        pred_file = os.path.splitext(image_path)[0]+pred
        
        with open(pred_file, 'r') as csvFile2:
            
            
            
            bbox = csv.reader(csvFile2, delimiter = " ")
            pred_per_image = []
            
            for row_pred in bbox:
                if not ''.join(row_pred).strip():
                   continue
                else:
                   pred_each = row_pred
                   #pred_each = np.asarray(row_pred, dtype=np.float).reshape(1,6)
                   pred_per_image.append(pred_each)
                   #print(pred_each)

        pred_per_image = np.asarray(pred_per_image, dtype=np.float)#.reshape(np.squeeze(pred_per_image, axis=1).shape)
            
        
        pred_per_image[:,1] = pred_per_image[:,1]*w
        pred_per_image[:,2] = pred_per_image[:,2]*h
        pred_per_image[:,3] = pred_per_image[:,3]*w
        pred_per_image[:,4] = pred_per_image[:,4]*h
           
            
            
               
        pred_square_per_image = np.asarray(copy.deepcopy(pred_per_image))
        #pred_square_per_image[:,0] = pred_per_image[:,0]
        pred_square_per_image[:,1] = np.round(pred_square_per_image[:,1]-pred_square_per_image[:,3]/2)
        pred_square_per_image[:,2] = np.round(pred_square_per_image[:,2]-pred_square_per_image[:,4]/2)
        pred_square_per_image[:,3] = np.round(pred_square_per_image[:,1]+pred_square_per_image[:,3]/2)
        pred_square_per_image[:,4] = np.round(pred_square_per_image[:,2]+pred_square_per_image[:,4]/2)
          
 

        squares = np.array([pred_square_per_image[:,1],pred_square_per_image[:,2], pred_square_per_image[:,3],pred_square_per_image[:,2], pred_square_per_image[:,3],pred_square_per_image[:,4], pred_square_per_image[:,1],pred_square_per_image[:,4]]).T
        squares = squares.reshape(-1,4,2)
        squares = squares.tolist()
            
        #np.concatenate((np.concatenate(([pred_square_per_image[:,1]].T,[pred_square_per_image[:,2]].T), axis=1), np.concatenate((pred_square_per_image[:,3],pred_square_per_image[:,2]), axis=1), np.concatenate((pred_square_per_image[:,3],pred_square_per_image[:,4]), axis=1), np.concatenate((pred_square_per_image[:,1],pred_square_per_image[:,4]), axis=1)), axis = 1)
        #np.array([[pred_square_per_image[:,:,1],pred_square_per_image[:,:,2]], [pred_square_per_image[:,:,3],pred_square_per_image[:,:,2]], [pred_square_per_image[:,:,3],pred_square_per_image[:,:,4]], [pred_square_per_image[:,:,1],pred_square_per_image[:,:,4]]])
        '''    
            
        image_rgb = cv2.imread(image_path)
        image_rgb = cv2.resize(image_rgb, (w,h))
        print(image_path)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
 
        print(image_index)
        pred_mosaic=[]
        
        if image_index == -1:
            print("this is execurted")
            global_mosaic = cv2.warpPerspective(image_rgb, offset_matrix, (row, col))
            
            #[pred_mosaic.append(cv2.perspectiveTransform(np.array(square).reshape((-1,1,2)), offset_matrix)) for square in squares]
                
            cor2.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), offset_matrix))
            
            '''
             #put prediction to global coordinate
            
            prediction_global = np.asarray(copy.deepcopy(pred_per_image))
            prediction_global[:,3] = np.round(pred_mosaic[:,2,0]-pred_mosaic[:,0,0])
            prediction_global[:,4] = np.round(pred_mosaic[:,2,1]-pred_mosaic[:,0,1])
            prediction_global[:,1] = np.round(pred_mosaic[:,0,0]+prediction_global[:,3]/2)
            prediction_global[:,2] = np.round(pred_mosaic[:,0,1]+prediction_global[:,4]/2)           
            '''
            
        else:
            wrapped = cv2.warpPerspective(image_rgb, np.dot(offset_matrix,(H_cum_new[image_index])), (row, col))
            
            #pred_mosaic.append(cv2.perspectiveTransform(np.array(square).reshape(-1,1,2)), np.dot(offset_matrix,(H_cum_new[image_index]))) for square in squares)
            
    
            (ret,data_map) = cv2.threshold(cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
            data_map = cv2.erode(data_map, np.ones((10,10), np.uint8))
            temp = cv2.add(global_mosaic, 0,  mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
            wrapped = cv2.add(wrapped, 0, mask = data_map, dtype=cv2.CV_8U)
            global_mosaic = cv2.add(temp, wrapped, dtype=cv2.CV_8U)
            cor2.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), np.dot(offset_matrix,(H_cum_new[image_index]))))
            
            '''
            #put prediction to global coordinate    
            prediction_global_temp = np.asarray(copy.deepcopy(pred_per_image))
            prediction_global_temp[:,3] = np.round(pred_mosaic[:,2,0]-pred_mosaic[:,0,0])
            prediction_global_temp[:,4] = np.round(pred_mosaic[:,2,1]-pred_mosaic[:,0,1])
            prediction_global_temp[:,1] = np.round(pred_mosaic[:,0,0]+prediction_global[:,3]/2)
            prediction_global_temp[:,2] = np.round(pred_mosaic[:,0,1]+prediction_global[:,4]/2)
            
            prediction_global = np.concatenate((prediction_global, prediction_global_temp),axis=0)
            '''
    
        if gps_initial and image_index >= 0:
            gps_projected = apply_homography_to_gps(gps_initial, np.dot(np.identity(3), H_cum_new[image_index]))
            print("Projected GPS for image %d: %s" % (image_index, str(gps_projected)))

            # Apply GPS to Pixel Projection only if transform exists (i.e., for .tiff files)
            if transform is not None:
                try:
                    pixel_coords = gps_to_pixel(gps_projected[0], gps_projected[1], transform)
                    print("Projected Pixel Coordinates for image %d: %s" % (image_index, str(pixel_coords)))
                except Exception as e:
                    print("Error in GPS to Pixel conversion: %s" % str(e))
            else:
                print("Skipping GPS to Pixel conversion for this image as no transform is available.")
        
				

        if image_index == stop - 1:
            gps_actual = extract_gps_from_image(image_path)
            print("Actual GPS from last image: %s" % str(gps_actual))						
        image_index += 1
        
        
        #display every new frame is added
        display_mosaic('mosaic_global_process', global_mosaic)
        


        if args.video == 'Y':
        
            #save each frame for mosaic
            save_mosaic[0:col, 0:row, :] = global_mosaic
            save_mosaic[col:col+h, row:row+w] = image_rgb
 
         
            save_mosaic = cv2.polylines(save_mosaic, np.int32([cor2[image_index]]), 1, (0,0,255), 10)
            save_mosaic = cv2.rectangle(save_mosaic, ( c-w, r-h), (c-10, r-10), (0,0,255), 10)
            save_mosaic = cv2.putText(save_mosaic, 'CurrentFrame_#{0:0=4d}'.format(image_index+1), (c-w, r-h-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 3, cv2.LINE_AA)

            cv2.waitKey(200) 
                   


            cv2.imwrite(save_path +"mosaic{:04d}.png".format(image_index), save_mosaic) 
            save_mosaic = save_mosaic-save_mosaic    

        
        if image_index == stop-1:
            cv2.imwrite(save_path + "global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png", global_mosaic)
            break
    '''
    if gps_projected and gps_actual:
        error_m = gps_error(gps_projected, gps_actual)
        print("Projected GPS:", gps_projected)
        print("Actual GPS:", gps_actual)
        print("GPS Error (meters):", error_m)


        #now lets draw the projected and the actual on global
        proj_x, proj_y = int(gps_projected[0]), int(gps_projected[1])
        cv2.circle(global_mosaic, (proj_x, proj_y), 10, (0, 255, 255), -1)


        if gps_actual:
            act_x, act_y = int(gps_actual[0]), int(gps_actual[1])
            cv2.rectangle(global_mosaic, (act_x-10, act_y-10), (act_x+10, act_y+10), (0, 0, 255), 3)
    '''

    save_mosaic_with_gps(global_mosaic, args.save_path, args, gps_projected, gps_actual, image_path)

    print(save_path)
    #cv2.imwrite(save_path, global_mosaic)
    cv2.imwrite(os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"_" + args.fname + ".png"), global_mosaic)
    #cv2.imwrite("result/corn_174_3fps_358/result_corn_174_all{:d}.png".format(image_index), (global_mosaic))
    


if __name__ == '__main__':

    start_time = datetime.now()
    main()
    elapsed = (datetime.now() - start_time).total_seconds()
    print('mosaicking time elapsed: ', elapsed)

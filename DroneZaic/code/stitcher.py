#!/usr/bin/env python

import os
import argparse
from datetime import datetime
import cv2
import csv
import numpy as np
from numpy.linalg import inv
import time
import sys
import copy
import exifread
from math import radians, cos, sin, sqrt, atan2
import rasterio
from rasterio.transform import rowcol, xy

np.set_printoptions(threshold=sys.maxsize)

def load_raster_transform(image_path):
    if image_path.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(image_path) as src:
                transform = src.transform
            print("DEBUG: Loaded raster transform for {}".format(image_path)) # Debug print
            return transform
        except Exception as e:
            print("ERROR: loading transform for .tiff file {}: {}".format(image_path, str(e))) # Debug print
            return None
    else:
        print("DEBUG: Skipping transform loading for non-tiff file: {}".format(image_path)) # Debug print
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
                print("DEBUG: Extracted GPS ({}, {}) from {}".format(lat, lon, os.path.basename(image_path))) # Debug print
    except Exception as e:
        print("ERROR: extracting GPS from {}: {}".format(os.path.basename(image_path), str(e))) # Debug print
    return gps

def gps_error(gps1, gps2):
    R = 6371000
    lat1, lon1 = radians(gps1[0]), radians(gps1[1])
    lat2, lon2 = radians(gps2[0]), radians(gps2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1 # Corrected from lon2 - lon2
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def apply_homography_to_gps(gps, H):
    x, y = gps
    pt = np.array([x, y, 1.0], dtype=np.float64)
    pt_proj = np.dot(H, pt)
    pt_proj = np.asarray(pt_proj).flatten()
    if pt_proj[2] == 0:
        print("WARNING: Homography resulted in zero for Z component, cannot project GPS accurately.")
        return (np.nan, np.nan)
    pt_proj /= pt_proj[2]
    return (pt_proj[0], pt_proj[1])

def save_mosaic_with_gps(global_mosaic, save_path, args, gps_projected=None, gps_actual=None, final_frame_path=None):
    print("DEBUG: Saving final mosaic and GPS info.") # Debug print
    if global_mosaic is None or global_mosaic.size == 0:
        print("ERROR: Global mosaic is empty or None. Cannot save.")
        return

    if gps_projected and gps_actual:
        error_m = gps_error(gps_projected, gps_actual)
        print("Projected GPS:", gps_projected)
        print("Actual GPS:", gps_actual)
        print("GPS Error (meters):", error_m)

        # Check if projected GPS pixels are within mosaic bounds before drawing
        proj_x, proj_y = int(gps_projected[0]), int(gps_projected[1])
        if 0 <= proj_x < global_mosaic.shape[1] and 0 <= proj_y < global_mosaic.shape[0]:
            cv2.circle(global_mosaic, (proj_x, proj_y), 10, (0, 255, 255), -1)
            print("DEBUG: Drawn projected GPS circle at ({}, {}).".format(proj_x, proj_y))
        else:
            print("WARNING: Projected GPS pixel ({}, {}) is outside mosaic bounds ({}x{}). Not drawing.".format(proj_x, proj_y, global_mosaic.shape[1], global_mosaic.shape[0]))

        # Actual GPS requires pixel conversion if transform is available
        if gps_actual and final_frame_path:
            transform_for_final_frame = None
            if final_frame_path.lower().endswith(('.tif', '.tiff')):
                transform_for_final_frame = load_raster_transform(final_frame_path)
            
            if transform_for_final_frame:
                act_x, act_y = gps_to_pixel(gps_actual[0], gps_actual[1], transform_for_final_frame)
                act_x, act_y = int(act_x), int(act_y) # Convert to int for drawing
                if 0 <= act_x < global_mosaic.shape[1] and 0 <= act_y < global_mosaic.shape[0]:
                    cv2.rectangle(global_mosaic, (act_x-10, act_y-10), (act_x+10, act_y+10), (0, 0, 255), 3)
                    print("DEBUG: Drawn actual GPS rectangle at ({}, {}).".format(act_x, act_y))
                else:
                    print("WARNING: Actual GPS pixel ({}, {}) is outside mosaic bounds ({}x{}). Not drawing.".format(act_x, act_y, global_mosaic.shape[1], global_mosaic.shape[0]))
            else:
                print("DEBUG: Cannot draw actual GPS on mosaic, no raster transform for final frame.")

    out_path = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + args.fname + ".png")
    cv2.imwrite(out_path, global_mosaic)
    print("Saved mosaic to:", out_path)

    if final_frame_path and gps_projected:
        pass # This part might need custom GeoTIFF handling, leaving as pass for now

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help='paths to one or more frames')
    parser.add_argument('-start', '--start', dest='start', default = 0, type=int, help="start stitching at")
    parser.add_argument('-stop', '--stop', dest='stop', default = 10000, type=int, help="stop stitching at")
    parser.add_argument('-save_path', dest='save_path', default="results/global_mosaic", type=str, help="path to save result")
    parser.add_argument('-r', '--rho', dest='r', default=10, type=int, help="directory value")
    parser.add_argument('-hm', '--homography', type=str, default=None, help='Homography Matrix to stitch from (CSV file)')
    parser.add_argument('-fname', '--fname', dest='fname', default='global_mosaic_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), help='desired filename for the global mosaic')
    parser.add_argument('-video', '--videos', dest = 'video', type=str, default= 'N', help='do you want to save frames addition process for videos?')
    parser.add_argument('-scale', '--scale', dest='scale', default=1, type=float, help='image size scale')
    parser.add_argument('-mini_mosaic', '--mini_mosaic', dest='mini_mosaic', action='store_true', help='enable mini mosaic')
    
    args = parser.parse_args()

    save_path = args.save_path
    print("DEBUG: Mini mosaic enabled: {}".format(args.mini_mosaic)) # Debug print
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("DEBUG: Created save path: {}".format(save_path)) # Debug print

    if args.mini_mosaic:
        mini_path = args.save_path
        if not os.path.exists(mini_path):
               os.makedirs(mini_path)
        save_path = mini_path

    all_img_files = []
    for p in args.image_path:
        if os.path.isdir(p):
            extensions = [".jpeg", ".jpg", ".png", ".JPG", ".tif", ".tiff"]
            print("DEBUG: Scanning directory for images: {}".format(p))
            found_files_in_dir = [os.path.join(p, file_name) for file_name in sorted(os.listdir(p), reverse=False) if os.path.splitext(file_name)[1].lower() in extensions]
            all_img_files.extend(found_files_in_dir)
            print("DEBUG: Found {} images in {}.".format(len(found_files_in_dir), p))
        elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpeg", ".jpg", ".png", ".JPG", ".tif", ".tiff"]:
            all_img_files.append(p)
            print("DEBUG: Added single image file: {}".format(p))
        else:
            print("WARNING: Provided path is not a directory or a supported image file: {}".format(p))
    
    if not all_img_files:
        print("ERROR: No image files found at the specified path(s). Exiting.")
        return

    frames_to_process = all_img_files[args.start : args.stop]
    print("DEBUG: Processing frames from index {} to {} (total: {} frames).".format(args.start, args.stop-1, len(frames_to_process)))
    
    if not frames_to_process:
        print("WARNING: No frames found in the range {} to {}. Exiting.".format(args.start, args.stop))
        return

    initial_img_path = frames_to_process[0]
    img = cv2.imread(initial_img_path)
    if img is None:
        print("ERROR: Could not load initial image {}. Exiting.".format(initial_img_path))
        return

    h_orig, w_orig, _ = img.shape
    w = int(np.round(w_orig / args.scale))
    h = int(np.round(h_orig / args.scale))
    print('DEBUG: Initial image path: {}'.format(initial_img_path))
    print('DEBUG: Original image dimensions (h, w): {}, {}'.format(h_orig, w_orig))
    print('DEBUG: Scaled image dimensions (h, w): {}, {}'.format(h, w))
    
    H_cum = []
    corners_h = []

    transform = None
    if initial_img_path.lower().endswith(('.tif', '.tiff')):
        transform = load_raster_transform(initial_img_path)
    
    gps_initial = extract_gps_from_image(initial_img_path)
    gps_projected = None
    gps_actual = None
    print("DEBUG: GPS of first image: {}".format(gps_initial))

    corners_4 = np.array([[0,0], [w,0],[w,h],[0,h]], dtype=np.float32)

    feature_detector = None
    bf_matcher = None

    if args.homography is None:
        print("DEBUG: No pre-computed homography matrix provided. Initiating feature-based stitching.")
        feature_detector = cv2.ORB_create(nfeatures=5000)
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_img_rgb_for_feature = None
    prev_img_gray_for_feature = None
    
    current_cumulative_H_local_feature = np.identity(3, dtype=np.float32) 

    print("\nDEBUG: Populating homographies for all frames...")
    for idx, current_img_path_for_h_comp in enumerate(frames_to_process):
        print("DEBUG: Processing image {}/{} for homography: {}".format(idx+1, len(frames_to_process), os.path.basename(current_img_path_for_h_comp)))
        
        current_img_rgb_for_h_comp = cv2.imread(current_img_path_for_h_comp)
        if current_img_rgb_for_h_comp is None:
            print("WARNING: Could not load image {} for homography computation. Skipping.".format(current_img_path_for_h_comp))
            continue
        
        current_img_rgb_resized = cv2.resize(current_img_rgb_for_h_comp, (w, h))
        current_img_gray_resized = cv2.cvtColor(current_img_rgb_resized, cv2.COLOR_RGB2GRAY)

        H_current_img_to_ref = None

        if args.homography is None:
            if idx == 0:
                H_current_img_to_ref = np.identity(3, dtype=np.float32)
                print("DEBUG: First image. Homography is Identity.")
            else:
                kp1, des1 = feature_detector.detectAndCompute(prev_img_gray_for_feature, None)
                kp2, des2 = feature_detector.detectAndCompute(current_img_gray_resized, None)

                print("DEBUG: Image {}: Previous keypoints: {}, Current keypoints: {}".format(idx, len(kp1) if kp1 is not None else 0, len(kp2) if kp2 is not None else 0))
                
                if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                    print("WARNING: Not enough keypoints/descriptors for image {}. Using previous cumulative homography (no change).".format(idx))
                    H_current_img_to_ref = current_cumulative_H_local_feature
                else:
                    matches = bf_matcher.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    print("DEBUG: Image {}: Found {} raw matches.".format(idx, len(matches)))

                    if len(matches) > 10:
                        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

                        H_curr_to_prev, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if H_curr_to_prev is None:
                            print("WARNING: Could not find homography for image {}. Using previous cumulative homography (no change).".format(idx))
                            H_current_img_to_ref = current_cumulative_H_local_feature
                        else:
                            print("DEBUG: Image {}: Successfully found homography.".format(idx))
                            H_current_img_to_ref = np.dot(current_cumulative_H_local_feature, H_curr_to_prev)
                            print("DEBUG: Image {}: Accumulated H (first->current):\n{}".format(idx, H_current_img_to_ref))
                    else:
                        print("WARNING: Not enough good matches ({}) for image {}. Using previous cumulative homography (no change).".format(len(matches), idx))
                        H_current_img_to_ref = current_cumulative_H_local_feature
            
            current_cumulative_H_local_feature = H_current_img_to_ref.copy()
            H_cum.append(H_current_img_to_ref)
            corners_h.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), H_current_img_to_ref))
            
            prev_img_rgb_for_feature = current_img_rgb_resized.copy()
            prev_img_gray_for_feature = current_img_gray_resized.copy()

        else: # If args.homography is not None (loading from CSV)
            # This block should be executed only once, before the loop, to populate H_cum.
            # If this is reached, it implies an error in the flow.
            # Assuming H_cum is correctly populated from CSV before this loop based on previous design.
            if idx < len(H_cum):
                 H_current_img_to_ref = H_cum[idx] # H_cum is already loaded from CSV
            else:
                 print("WARNING: Homography matrix for image {} not found in CSV. Using identity.".format(idx))
                 H_current_img_to_ref = np.identity(3, dtype=np.float32)
                 H_cum.append(H_current_img_to_ref)
                 corners_h.append(cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), H_current_img_to_ref))


    print("DEBUG: Homography population complete.")
    
    if not corners_h:
        print("ERROR: No corners information available for mosaic dimensions. Exiting.")
        return

    corners_h_arr = np.asarray(corners_h)
    max_x = np.max(corners_h_arr[...,0].flatten())
    min_x = np.min(corners_h_arr[...,0].flatten())
    max_y = np.max(corners_h_arr[...,1].flatten())
    min_y = np.min(corners_h_arr[...,1].flatten())

    print("DEBUG: Calculated raw min/max X: ({}, {}), Y: ({}, {})".format(min_x, max_x, min_y, max_y))

    offset_x = 0
    offset_y = 0
    if min_x < 0:
        offset_x = np.ceil(-(min_x))
    if min_y < 0:
        offset_y = np.ceil(-(min_y))
    
    print("DEBUG: Calculated offsets: offset_x={}, offset_y={}".format(offset_x, offset_y))

    final_mosaic_width = int(np.floor(max_x + offset_x))
    final_mosaic_height = int(np.floor(max_y + offset_y))

    if (final_mosaic_width % 2) != 0: final_mosaic_width += 1
    if (final_mosaic_height % 2) != 0: final_mosaic_height += 1
    
    print("DEBUG: Final mosaic dimensions: {}x{}".format(final_mosaic_width, final_mosaic_height))

    final_offset_matrix = np.identity(3, np.float32)
    final_offset_matrix[0,2] = offset_x
    final_offset_matrix[1,2] = offset_y
    print("DEBUG: Final offset matrix:\n{}".format(final_offset_matrix))

    global_mosaic = np.zeros((final_mosaic_height, final_mosaic_width, 3), np.uint8)
    print("DEBUG: Initialized global mosaic canvas of size {}.".format(global_mosaic.shape))

    print("\nDEBUG: Warping and blending images onto final mosaic...")
    for idx, current_img_path_for_warp in enumerate(frames_to_process):
        print("DEBUG: Warping image {}/{}: {}".format(idx+1, len(frames_to_process), os.path.basename(current_img_path_for_warp)))

        if idx >= len(H_cum):
            print("WARNING: No homography found for image {} during final warp pass. Skipping warp.".format(idx))
            continue

        image_rgb = cv2.imread(current_img_path_for_warp)
        if image_rgb is None:
            print("ERROR: Could not load image {} for final warp. Skipping.".format(current_img_path_for_warp))
            continue
        
        image_rgb = cv2.resize(image_rgb, (w, h))

        H_image_to_ref = H_cum[idx]

        H_total = np.dot(final_offset_matrix, H_image_to_ref)
        print("DEBUG: Image {}: H_total (offset * H_image_to_ref):\n{}".format(idx, H_total))

        warped_image = cv2.warpPerspective(image_rgb, H_total, (final_mosaic_width, final_mosaic_height))
        print("DEBUG: Image {}: Warped image shape: {}".format(idx, warped_image.shape))

        (ret, data_map) = cv2.threshold(cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
        data_map = cv2.erode(data_map, np.ones((10,10), np.uint8))
        
        temp_global_part = cv2.add(global_mosaic, 0, mask=np.bitwise_not(data_map))
        new_image_part = cv2.add(warped_image, 0, mask=data_map)
        
        global_mosaic = cv2.add(temp_global_part, new_image_part)
        print("DEBUG: Image {}: Blended into global mosaic.".format(idx))
        
        if idx == len(frames_to_process) - 1:
            print("DEBUG: Processing GPS for the last image.")
            if gps_initial:
                gps_projected = apply_homography_to_gps(gps_initial, H_total)
                print("DEBUG: Projected GPS of original reference point for last image's transformation: {}".format(str(gps_projected)))
            
            gps_actual = extract_gps_from_image(current_img_path_for_warp)
            print("DEBUG: Actual GPS from last processed image: {}".format(str(gps_actual)))

    final_image_for_gps_path = frames_to_process[-1] if frames_to_process else None
    save_mosaic_with_gps(global_mosaic, save_path, args, gps_projected, gps_actual, final_image_for_gps_path)
    
if __name__ == '__main__':
    start_time = datetime.now()
    print("DEBUG: Script started at {}".format(start_time))
    main()
    elapsed = (datetime.now() - start_time).total_seconds()
    print('DEBUG: Mosaicking time elapsed: {} seconds.'.format(elapsed))
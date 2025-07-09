#!/usr/bin/env python VERION 4 image stitch OPENCV

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
    row, col = rowcol(transform, gps_lon, gps_lat) # Corrected 'lat' to 'gps_lat'
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
    # This function might become less relevant if GPS projection is handled differently
    # or not needed in the stitched output. Keeping for now.
    x, y = gps
    pt = np.array([x, y, 1.0], dtype=np.float64)
    pt_proj = np.dot(H, pt)
    pt_proj = np.asarray(pt_proj).flatten()
    pt_proj /= pt_proj[2]
    return (pt_proj[0], pt_proj[1])

def save_mosaic_with_gps(global_mosaic, save_path, args, gps_projected=None, gps_actual=None, final_frame_path=None):
    # GPS features drawing will depend on how GPS points are mapped to the final mosaic
    # by cv2.Stitcher. This part might need significant re-evaluation.
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
    # cv2.imshow('test test', np.uint8(img)) # Commented out to prevent display issues
    # cv2.waitKey(100) # Commented out to prevent display issues

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help='paths to one or more frames')
    parser.add_argument('-start', '--start', dest='start', default = 0, type=int, help="start stitching at")
    parser.add_argument('-stop', '--stop', dest='stop', default = 10000, type=int, help="stop stitching at")
    parser.add_argument('-save_path', dest='save_path', default="results/global_mosaic", type=str, help="path to save result")
    parser.add_argument('-r', '--rho', dest='r', default=10, type=int, help="directory value")
    # The homography argument is no longer needed for cv2.Stitcher direct use
    # parser.add_argument('-hm', '--homography', type=str, help='txt or csv file that stores homography matrices')
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
        mini_path = args.save_path
        if not os.path.exists(mini_path):
             os.makedirs(mini_path)
        save_path = mini_path

    # Removed unused variables from old stitching logic
    # result = None
    # result_gry = None
    # H_cum = []
    # H=[]
    # cor2 = []
    # temp_c_normalized = np.zeros((3,4))

    stop = args.stop # Stop argument might not be directly applicable for cv2.Stitcher, it will stitch all provided images
    # homography_matrix = args.homography # No longer used
    
    # Adjusted image loading to explicitly read images into a list for cv2.Stitcher
    all_image_paths_in_group = []
    if args.image_path and os.path.isdir(args.image_path[0]): # Expecting a single directory path
        image_dir = args.image_path[0]
        extensions = [".jpeg", ".jpg", ".png", ".JPG", ".tif", ".tiff"]
        for file_name in sorted(os.listdir(image_dir), reverse=False):
            if os.path.splitext(file_name)[1].lower() in extensions:
                all_image_paths_in_group.append(os.path.join(image_dir, file_name))
    else:
        print("ERROR: Please provide a valid directory for images when using cv2.Stitcher.", file=sys.stderr)
        sys.exit(1)

    if not all_image_paths_in_group:
        print("ERROR: No images found in the specified directory. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Load all images for the current mini-mosaic group
    images_to_stitch = []
    initial_img_path = all_image_paths_in_group[0] # First image to get dimensions and GPS
    
    # Get dimensions from the first image
    first_img_raw = cv2.imread(initial_img_path)
    if first_img_raw is None:
        print(f"ERROR: Could not read the first image: {initial_img_path}. Exiting.", file=sys.stderr)
        sys.exit(1)

    h_orig, w_orig, _ = first_img_raw.shape
    w = int(np.round(w_orig / args.scale))
    h = int(np.round(h_orig / args.scale))
    print(f'Original image dimensions (h, w): {h_orig} {w_orig}', file=sys.stderr)
    print(f'Scaled image dimensions (h, w): {h} {w}', file=sys.stderr)


    for img_path in all_image_paths_in_group:
        img_read = cv2.imread(img_path)
        if img_read is None:
            print(f"WARNING: Could not read image {img_path}. Skipping it for stitching.", file=sys.stderr)
            continue
        # Resize images if scale is not 1.0. Stitcher works best with consistent input sizes.
        if args.scale != 1.0:
            img_read = cv2.resize(img_read, (w, h))
        images_to_stitch.append(img_read)

    if not images_to_stitch:
        print("ERROR: No images loaded successfully for stitching. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Attempting to stitch {len(images_to_stitch)} images using cv2.Stitcher.", file=sys.stderr)

    # --- CV2.Stitcher Integration ---
    # <<<<< START OF CHANGE >>>>>
    # Removed direct setter calls for 'setFeaturesFinder' and 'setBlender'
    # as they caused 'AttributeError'.
    # We will rely on default Stitcher behavior or explore other configuration methods if needed.
    stitcher = cv2.Stitcher_create()

    # The problematic lines are commented out or removed.
    # stitcher.setFeaturesFinder(cv2.ORB_create(nfeatures=25000))
    # stitcher.setBlender(cv2.detail.FeatherBlender())
    # stitcher.setWarper(cv2.PyRotationWarper("plane", 1))
    # <<<<< END OF CHANGE >>>>>

    status, global_mosaic = stitcher.stitch(images_to_stitch)

    # The status codes are now directly integers returned by the stitcher.stitch method
    # and not attributes of the cv2.Stitcher class directly in newer OpenCV versions.
    # We compare against the numerical values which are 0 for OK, 1 for ERR_NEED_MORE_IMGS etc.
    # Note: These integer values are consistent, but looking up the exact values in OpenCV docs
    # or printing `cv2.Stitcher_OK` (if available in your version) for verification is good practice.
    if status == 0: # cv2.Stitcher.OK is typically 0
        print("Stitching successful!", file=sys.stderr)
    elif status == 1: # cv2.Stitcher.ERR_NEED_MORE_IMGS is typically 1
        print("Stitching failed: Need more images or insufficient overlap/features.", file=sys.stderr)
        sys.exit(1)
    elif status == 2: # cv2.Stitcher.ERR_HOMOGRAPHY_EST_FAIL is typically 2
        print("Stitching failed: Homography estimation failed. Check image quality/overlap.", file=sys.stderr)
        sys.exit(1)
    elif status == 3: # cv2.Stitcher.ERR_CAMERA_PARAMS_ADJUST_FAIL is typically 3
        print("Stitching failed: Camera parameters adjustment failed.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Stitching failed with unexpected error code: {status}", file=sys.stderr)
        sys.exit(1)
    # --- END CV2.Stitcher Integration ---

    # The rest of the script needs to adapt to `global_mosaic` being the direct output
    # The old `row,col,channel = global_mosaic.shape` and offset matrix calculations are now handled by Stitcher
    # GPS projection logic might need to be re-evaluated as Stitcher's coordinate system is internal.

    # GPS extraction for the first image (as a reference point for original location)
    gps_initial = extract_gps_from_image(initial_img_path)
    # gps_projected and gps_actual logic is complex with cv2.Stitcher as it doesn't expose internal H matrices directly
    # For initial testing, we might skip the GPS visualization part in save_mosaic_with_gps
    gps_projected = None
    gps_actual = None

    # The `video` argument for saving individual frames of the stitching process
    # is not directly compatible with cv2.Stitcher's atomic operation.
    # If this feature is critical, it would require a more complex re-implementation
    # or finding a way to integrate with Stitcher's intermediate steps.
    if args.video == 'Y':
        print("WARNING: 'video' saving is not directly supported with cv2.Stitcher's atomic stitching.", file=sys.stderr)

    save_mosaic_with_gps(global_mosaic, args.save_path, args, gps_projected, gps_actual, all_image_paths_in_group[-1])

    print("Final mosaic saved to: " + save_path, file=sys.stderr)
    cv2.imwrite(os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"_" + args.fname + ".png"), global_mosaic)


if __name__ == '__main__':

    start_time = datetime.now()
    main()
    elapsed = (datetime.now() - start_time).total_seconds()
    print('mosaicking time elapsed: ', elapsed)
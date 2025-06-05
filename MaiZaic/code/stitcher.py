#!/usr/bin/env python

import os
import argparse
from datetime import datetime
import cv2
import csv
import numpy as np
from numpy.linalg import inv # Keep inv in case it's needed for other H definitions in future
import time
import sys

# np.set_printoptions(threshold=sys.maxsize) # Usually for debugging, can be commented out

def display_mosaic(fname, img):
    # Ensure img is not empty and has valid dimensions
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        print "Warning (display_mosaic): Image '{}' is empty or has invalid dimensions.".format(fname)
        return

    max_display_dim = 1200
    h, w = img.shape[:2]
    
    scale = 1.0
    if h > max_display_dim or w > max_display_dim:
        scale_h = float(max_display_dim) / h if h > 0 else 1.0
        scale_w = float(max_display_dim) / w if w > 0 else 1.0
        scale = min(scale_h, scale_w)

    shape = (int(scale * w), int(scale * h))
    
    if shape[0] == 0 and w > 0: shape = (1, shape[1])
    if shape[1] == 0 and h > 0: shape = (shape[0], 1)
    if shape[0] <=0 or shape[1] <=0:
        print "Warning (display_mosaic): Cannot display image '{}' due to invalid scaled dimensions {}.".format(fname, shape)
        return

    try:
        img_display = cv2.resize(img, shape, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        # commented since its being ran in a docker container cv2.imshow(str(fname), np.uint8(img_display)) # Window name must be a string
        cv2.waitKey(100) # Adjust waitKey as needed
    except cv2.error as e:
        print "OpenCV Error in display_mosaic for '{}': {}".format(fname, e)
    except Exception as e:
        print "General Error in display_mosaic for '{}': {}".format(fname, e)


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help='paths to one or more frames or directories containing frames')
    parser.add_argument('-start', '--start', dest='start', default=0, type=int, help="start stitching at (currently unused in revised logic)")
    parser.add_argument('-stop', '--stop', dest='stop', default=10000, type=int, help="stop stitching after this many images")
    parser.add_argument('-save_path', dest='save_path', default="results/global_mosaic", type=str, help="path to save result")
    parser.add_argument('-hm', '--homography', type=str, required=True, help='txt or csv file that stores homography matrices')
    parser.add_argument('-fname', '--fname', dest='fname', default='stitched_mosaic', help='desired base filename for the global mosaic')
    parser.add_argument('-video', '--videos', dest = 'video', type=str, default= 'N', help='do you want to save frames addition process for videos? (Y/N)')
    parser.add_argument('-mini_mosaic', '--mini_mosaic', dest='mini_mosaic', action='store_true', help='enable mini mosaic (affects save_path)')

    args = parser.parse_args()

    save_path = args.save_path
    if args.mini_mosaic:
        mini_path = args.save_path 
        if not os.path.exists(mini_path):
             os.makedirs(mini_path)
        save_path = mini_path 
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print "Created save directory: {}".format(save_path)

    homography_matrix_file = args.homography

    image_files_list = []
    if not args.image_path:
        print "Error: -image_path argument is required."
        sys.exit(1)
        
    for image_path_arg in args.image_path:
        if not os.path.exists(image_path_arg):
            print "Warning: Provided image path does not exist: {}".format(image_path_arg)
            continue
        if os.path.isdir(image_path_arg):
            extensions = [".jpeg", ".jpg", ".png", "JPG"]
            try:
                for file_p in sorted(os.listdir(image_path_arg), reverse=False):
                    if os.path.splitext(file_p)[1].lower() in extensions:
                        image_files_list.append(os.path.join(image_path_arg, file_p))
            except OSError as e:
                print "Error reading directory {}: {}".format(image_path_arg, e)
                continue
        else: 
            image_files_list.append(image_path_arg)

    if not image_files_list:
        print "Error: No image files found to process. Please check -image_path arguments."
        sys.exit(1)
    
    print "Found {} image(s) to process.".format(len(image_files_list))

    try:
        img_ref = cv2.imread(image_files_list[0])
        if img_ref is None:
            print "Error: Could not read the first image: {}".format(image_files_list[0])
            sys.exit(1)
        h_ref, w_ref, _ = img_ref.shape
        print "Reference image dimensions (H x W): {} x {}".format(h_ref, w_ref)
    except IndexError:
        print "Error: image_files_list is empty, cannot get reference dimensions."
        sys.exit(1)
    except Exception as e:
        print "Error getting reference image dimensions: {}".format(e)
        sys.exit(1)

    H_raw_list = []
    try:
        with open(homography_matrix_file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=",")
            for i, row_data in enumerate(reader):
                if not row_data or not ''.join(row_data).strip():
                    print "Warning: Skipping empty or blank row {} in homography file.".format(i+1)
                    continue
                try:
                    H_each = np.asarray(row_data, dtype=float).reshape(3,3)
                    H_raw_list.append(H_each)
                except ValueError as e:
                    print "Error parsing row {} in homography file: {}. Error: {}. Skipping this homography.".format(i+1, row_data, e)
    except IOError: # Changed from FileNotFoundError for Python 2.7
        print "Error: Homography file not found: {}".format(homography_matrix_file)
        sys.exit(1)
    except Exception as e:
        print "Error reading homography file {}: {}".format(homography_matrix_file, e)
        sys.exit(1)
        
    print "Read {} homography matrices.".format(len(H_raw_list))

    H_transforms_to_world = []
    num_expected_images = len(H_raw_list) + 1

    if num_expected_images > 0:
        current_H_to_world = np.identity(3)
        H_transforms_to_world.append(np.copy(current_H_to_world))

        for i in range(num_expected_images - 1):
            if i < len(H_raw_list):
                current_H_to_world = np.dot(current_H_to_world, H_raw_list[i])
                H_transforms_to_world.append(np.copy(current_H_to_world))
            else:
                print "Warning: Missing homography for image index {}. Using last known transform.".format(i+1)
                H_transforms_to_world.append(np.copy(H_transforms_to_world[-1]))

    all_corners_in_world_list = []
    img_corners_local = np.array([[1,1], [w_ref,1], [w_ref,h_ref], [1,h_ref]], dtype=np.float32).reshape((-1,1,2))

    for k_idx in range(len(H_transforms_to_world)):
        H_k_to_world = H_transforms_to_world[k_idx]
        transformed_k_corners = cv2.perspectiveTransform(img_corners_local, H_k_to_world)
        
        if transformed_k_corners is not None:
            all_corners_in_world_list.extend(transformed_k_corners)
        else:
            print "Warning: cv2.perspectiveTransform returned None for image index {} with H: {}".format(k_idx, H_k_to_world)
            all_corners_in_world_list.extend(img_corners_local)

    if not all_corners_in_world_list:
        print "Error: No transformed corners to calculate canvas size. Using default size of one image."
        min_x, max_x, min_y, max_y = 1.0, float(w_ref), 1.0, float(h_ref)
        if num_expected_images == 0 and len(image_files_list) == 0:
            print "Error: No images to process."
            sys.exit(1)
    else:
        all_corners_np_arr = np.array(all_corners_in_world_list)
        valid_corners = all_corners_np_arr[np.isfinite(all_corners_np_arr).all(axis=(1,2))]
        if valid_corners.shape[0] == 0:
            print "Error: All transformed corners are NaN/Inf. Check homographies. Using default size."
            min_x, max_x, min_y, max_y = 1.0, float(w_ref), 1.0, float(h_ref)
        else:
            min_x = np.min(valid_corners[:,0,0])
            max_x = np.max(valid_corners[:,0,0])
            min_y = np.min(valid_corners[:,0,1])
            max_y = np.max(valid_corners[:,0,1])

    offset_x = 0.0
    offset_y = 0.0
    if min_x < 1.0:
        offset_x = np.ceil(1.0 - min_x)
    if min_y < 1.0:
        offset_y = np.ceil(1.0 - min_y)

    canvas_width = int(np.ceil(max_x - min_x + (offset_x if min_x < 1.0 else 0) ) )
    canvas_height = int(np.ceil(max_y - min_y + (offset_y if min_y < 1.0 else 0) ) )
    
    if canvas_width <= 0: canvas_width = w_ref
    if canvas_height <= 0: canvas_height = h_ref
    
    offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    print "Calculated min/max extents (x_min, x_max, y_min, y_max): {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(min_x, max_x, min_y, max_y)
    print "Calculated canvas size (W x H): {} x {}".format(canvas_width, canvas_height)
    print "Offset matrix:\n{}".format(offset_matrix)

    try:
        # Calculate GB size carefully for Python 2 division
        gb_size = (float(canvas_height) * canvas_width * 3) / (1024.0**3)
        global_mosaic = np.zeros((canvas_height, canvas_width, 3), np.uint8)
    except MemoryError:
        print "MemoryError: Cannot allocate mosaic of size {} x {} ({:.2f} GB)".format(canvas_height, canvas_width, gb_size)
        sys.exit(1)
    except ValueError as e: 
        print "ValueError: Cannot allocate mosaic, likely due to invalid dimensions {} x {}. Error: {}".format(canvas_height, canvas_width, e)
        sys.exit(1)

    canvas_rows, canvas_cols = canvas_height, canvas_width

    cor2_for_video = []
    num_images_to_process = min(len(image_files_list), len(H_transforms_to_world), args.stop)

    for k_img_idx in range(num_images_to_process):
        current_image_file_path = image_files_list[k_img_idx]
        print "Processing image {}/{}: {}".format(k_img_idx + 1, num_images_to_process, current_image_file_path)
        
        image_rgb = cv2.imread(current_image_file_path)
        if image_rgb is None:
            print "Error: Could not read image {}. Skipping.".format(current_image_file_path)
            continue
        
        if image_rgb.shape[0] != h_ref or image_rgb.shape[1] != w_ref:
            print "Warning: Image {} dimensions ({}x{}) differ from reference ({}x{}). Resizing.".format(current_image_file_path, image_rgb.shape[1], image_rgb.shape[0], w_ref, h_ref)
            image_rgb = cv2.resize(image_rgb, (w_ref, h_ref))

        transform_for_current_image = H_transforms_to_world[k_img_idx]
        final_warp_matrix = np.dot(offset_matrix, transform_for_current_image)
        
        warped_image = cv2.warpPerspective(image_rgb, final_warp_matrix, (canvas_cols, canvas_rows))
        
        (ret, data_map_for_blend) = cv2.threshold(cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
        erosion_kernel = np.ones((10,10), np.uint8)
        data_map_for_blend = cv2.erode(data_map_for_blend, erosion_kernel)

        temp_mosaic_preserved_parts = cv2.bitwise_and(global_mosaic, global_mosaic, mask=cv2.bitwise_not(data_map_for_blend))
        warped_image_visible_parts = cv2.bitwise_and(warped_image, warped_image, mask=data_map_for_blend)
        global_mosaic = cv2.add(temp_mosaic_preserved_parts, warped_image_visible_parts)

        if args.video == 'Y':
            corners_transformed_for_video = cv2.perspectiveTransform(img_corners_local, final_warp_matrix)
            if corners_transformed_for_video is not None:
                 cor2_for_video.append(corners_transformed_for_video)

        display_mosaic('Mosaic_Stitching_Process_Frame_{}'.format(k_img_idx+1), global_mosaic)

        if args.video == 'Y':
            display_frame_for_video = global_mosaic.copy()
            if k_img_idx < len(cor2_for_video) and cor2_for_video[k_img_idx] is not None:
                 cv2.polylines(display_frame_for_video, [np.int32(cor2_for_video[k_img_idx])], True, (0,0,255), 3, cv2.LINE_AA)
            
            video_frame_save_path = os.path.join(save_path, "video_frames")
            if not os.path.exists(video_frame_save_path):
                os.makedirs(video_frame_save_path)
            cv2.imwrite(os.path.join(video_frame_save_path, "mosaic_frame_{:04d}.png".format(k_img_idx)), display_frame_for_video)

    final_save_name = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + args.fname + ".png")
    try:
        cv2.imwrite(final_save_name, global_mosaic)
        print "Final mosaic saved to: {}".format(final_save_name)
    except cv2.error as e:
        print "Error saving final mosaic {}: {}".format(final_save_name, e)
    except Exception as e:
        print "An unexpected error occurred while saving {}: {}".format(final_save_name, e)

    end_time = time.time()
    print "Time elapsed: {:.2f} seconds".format(end_time - start_time)
    cv2.destroyAllWindows()
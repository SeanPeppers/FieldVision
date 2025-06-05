'''
project: MaiZaic

author: Dewi Kharismawati

about this script:
    - this is surf_assembly_global.py
    - this is to create a global mosaic for all minimosaic using asift (now surf)
    - result will be png file of global mosaic
    - homography between mini mosaic also will be save in the save_path

to call:
    python surf_assembly_global.py -image_path /path/to/mini/mosaic -save_path /path/to/save

'''

import os
import argparse
from datetime import datetime
import cv2
import csv
import numpy as np
from numpy.linalg import inv
from surf import surf # Assuming this is your SURF-based homography estimation
import time
import sys

OPENCV_DIM_LIMIT = 32766 # OpenCV's typical dimension limit for warpPerspective source/destination

def mosaicking(img0, img1, counter, h_all, H_tp_ignored):
    """
    Stitches a new image (img0) onto an existing mosaic (img1) using a homography.

    Args:
        img0 (np.ndarray): The new image (mini-mosaic) to be added.
        img1 (np.ndarray): The current accumulated global mosaic.
        counter (int): Current stitch counter (for logging/debug).
        h_all (np.ndarray): Homography matrix from img1 to img0 (H_current_global_to_new_mini).
        H_tp_ignored (np.ndarray): Placeholder, unused in current logic.

    Returns:
        tuple: (output_img, H_translation) where output_img is the new global mosaic,
               and H_translation is the translation matrix applied.
               Returns (img1, np.identity(3)) if stitching fails or is skipped.
    """
    print "DEBUG (mosaicking): Adding new frame. Counter: {}".format(counter)
    
    # h_all is H_img1_to_img0. We need H_img0_to_img1 to warp img0.
    try:
        h_for_warp = inv(h_all)
        # Check for non-finite values in the inverted matrix
        if not np.isfinite(h_for_warp).all():
            print "ERROR (mosaicking): Inverted homography contains NaN or Inf. H_all was:\n{}".format(h_all)
            return img1, np.identity(3)
    except np.linalg.LinAlgError as e:
        print "ERROR (mosaicking): Could not invert homography matrix: {}".format(e)
        print "H_all was:\n{}".format(h_all)
        return img1, np.identity(3)

    # Define corners of the images
    points0 = np.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2)) # Corners of the new mini-mosaic (img0)
    
    points1 = np.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2)) # Corners of the current global mosaic (img1)
    
    try:
        # Transform img0's corners into img1's coordinate system
        points2 = cv2.perspectiveTransform(points0, h_for_warp)
        if points2 is None or not np.isfinite(points2).all():
            print "ERROR (mosaicking): cv2.perspectiveTransform returned None or non-finite points for points2. Skipping this image."
            return img1, np.identity(3)
    except cv2.error as e:
        print "ERROR (mosaicking): cv2.perspectiveTransform for points2 failed: {}".format(e)
        return img1, np.identity(3)

    # Combine all corners to find the new bounding box
    points = np.concatenate((points1, points2), axis=0)
    
    # Filter out any non-finite points before calculating bounds
    finite_points = points[np.isfinite(points).all(axis=(1,2))]

    if finite_points.shape[0] < 1: # Need at least one valid point to determine bounds
        print "WARNING (mosaicking): No finite points after combining original and transformed corners. Skipping."
        return img1, np.identity(3)

    # Calculate min/max coordinates for the new canvas
    x_min, y_min = np.int32(finite_points.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(finite_points.max(axis=0).ravel() + 0.5)

    # Calculate translation to shift all coordinates to be positive
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    
    # Calculate the dimensions of the new canvas
    warp_canvas_width = x_max - x_min
    warp_canvas_height = y_max - y_min

    print "DEBUG (mosaicking): Calculated warp canvas dimensions (W x H): {} x {}".format(warp_canvas_width, warp_canvas_height)
    print "DEBUG (mosaicking): img0 (new mini-mosaic) dimensions (W x H): {} x {}".format(img0.shape[1], img0.shape[0])
    print "DEBUG (mosaicking): img1 (current global mosaic) dimensions (W x H): {} x {}".format(img1.shape[1], img1.shape[0])

    # Robustness checks for canvas dimensions
    if warp_canvas_width <= 0 or warp_canvas_height <= 0 :
        print "ERROR (mosaicking): Invalid calculated canvas dimensions (W={}, H={}). Skipping image.".format(warp_canvas_width, warp_canvas_height)
        return img1, np.identity(3)

    if warp_canvas_width > OPENCV_DIM_LIMIT or \
       warp_canvas_height > OPENCV_DIM_LIMIT :
        print "ERROR (mosaicking): Calculated canvas dimensions ({0}x{1}) exceed OpenCV limit of {2} pixels. Max possible GB: {3:.2f}".format(
            warp_canvas_width, warp_canvas_height, OPENCV_DIM_LIMIT, (float(warp_canvas_width) * warp_canvas_height * 3) / (1024.0**3)
        )
        print "Skipping this image due to canvas size constraints."
        return img1, np.identity(3) # Return original mosaic if canvas too big

    try:
        # Create the new larger canvas
        output_img = np.zeros((warp_canvas_height, warp_canvas_width, 3), dtype=np.uint8)
    except MemoryError:
        print "ERROR (mosaicking): MemoryError allocating canvas of size {}x{}. Returning original mosaic.".format(warp_canvas_height, warp_canvas_width)
        return img1, np.identity(3)

    # Place the existing mosaic (img1) onto the new canvas
    # The slicing indices are relative to the new canvas's origin (0,0) after translation
    # Ensure indices are within bounds of the output_img
    y_start_img1_on_new_canvas = -y_min
    y_end_img1_on_new_canvas = img1.shape[0] - y_min
    x_start_img1_on_new_canvas = -x_min
    x_end_img1_on_new_canvas = img1.shape[1] - x_min

    # Clamp indices to ensure they fit within the new output_img bounds
    y_start_img1_clamped = max(0, y_start_img1_on_new_canvas)
    y_end_img1_clamped = min(output_img.shape[0], y_end_img1_on_new_canvas)
    x_start_img1_clamped = max(0, x_start_img1_on_new_canvas)
    x_end_img1_clamped = min(output_img.shape[1], x_end_img1_on_new_canvas)

    if (y_start_img1_clamped >= y_end_img1_clamped or \
        x_start_img1_clamped >= x_end_img1_clamped):
        print "WARNING (mosaicking): Current global mosaic (img1) is completely off-canvas or has zero dimensions after clamping. It will not be placed."
    else:
        # Calculate the corresponding slice from img1 to copy to output_img
        # This handles cases where img1 might partially go off the new canvas
        img1_slice_y_start = max(0, -y_start_img1_on_new_canvas)
        img1_slice_y_end = img1_slice_y_start + (y_end_img1_clamped - y_start_img1_clamped)
        img1_slice_x_start = max(0, -x_start_img1_on_new_canvas)
        img1_slice_x_end = img1_slice_x_start + (x_end_img1_clamped - x_start_img1_clamped)

        output_img[y_start_img1_clamped:y_end_img1_clamped, \
                   x_start_img1_clamped:x_end_img1_clamped] = \
                   img1[img1_slice_y_start:img1_slice_y_end, \
                        img1_slice_x_start:img1_slice_x_end]
    
    # Calculate the final transformation matrix for img0 (new mini-mosaic)
    # This matrix transforms img0 directly to its position on the new, offset canvas
    final_warp_H = np.dot(H_translation, h_for_warp)

    try:
        # Warp img0 (new mini-mosaic) onto the new canvas
        warped_img = cv2.warpPerspective(img0, final_warp_H, (warp_canvas_width, warp_canvas_height))
    except cv2.error as e:
        print "ERROR (mosaicking): cv2.warpPerspective failed for img0: {}".format(e)
        print "img0 shape: {}, final_warp_H:\n{}, canvas_size: ({},{})".format(img0.shape, final_warp_H, warp_canvas_width, warp_canvas_height)
        return img1, np.identity(3) # Return original mosaic

    # Blending (simple binary mask blending)
    # Create a mask from the warped image to identify its non-black pixels
    mask2 = (warped_img > 0).astype(np.uint8) * 255 
    # Erode the mask to remove potential artifacts at the edges
    mask3 = cv2.erode(mask2, np.ones((10,10), np.uint8))

    # Mask out the region where the new image will be placed in the existing mosaic
    masked_mosaic_bg = cv2.bitwise_and(output_img, output_img, mask=cv2.bitwise_not(mask3))
    # Keep only the valid parts of the warped new image
    warped_img_fg = cv2.bitwise_and(warped_img, warped_img, mask=mask3)
    
    # Combine the masked background of the old mosaic with the foreground of the new warped image
    output_img = cv2.bitwise_or(warped_img_fg, masked_mosaic_bg)

    return output_img, H_translation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', required=True, help="paths to one or more mini-mosaic images or a directory containing them")
    parser.add_argument('-save_path', dest='save_path', default="global_mosaic_surf/", type=str, help="path to save result")

    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print "Created save directory: {}".format(save_path)

    result = None # Stores the accumulated global mosaic
    result_gry = None # Grayscale version of result for feature matching

    all_mini_mosaic_paths = []
    if args.image_path:
        for path_arg in args.image_path:
            if os.path.isdir(path_arg):
                extensions = [".jpeg", ".jpg", ".png", "JPG"] # Added JPG for consistency
                try:
                    for file_name in sorted(os.listdir(path_arg)):
                        if os.path.splitext(file_name)[1].lower() in extensions:
                            all_mini_mosaic_paths.append(os.path.join(path_arg, file_name))
                except OSError as e:
                    print "Error reading directory {}: {}".format(path_arg, e)
            elif os.path.isfile(path_arg):
                all_mini_mosaic_paths.append(path_arg)
            else:
                print "Warning: Path {} is not a valid file or directory.".format(path_arg)
    
    if not all_mini_mosaic_paths:
        print "Error: No mini-mosaic images found to process from path(s): {}".format(args.image_path)
        sys.exit(1)
    
    print "Found {} mini-mosaics to assemble.".format(len(all_mini_mosaic_paths))

    H_tp_dummy = np.identity(3) # This is a placeholder, as the return value of mosaicking is not fully used.
    successful_stitches = 0

    for image_index, current_mini_mosaic_path in enumerate(all_mini_mosaic_paths):
        print "\n--- Processing mini-mosaic ({} of {}): {} ---".format(image_index + 1, len(all_mini_mosaic_paths), current_mini_mosaic_path)
        
        image_color = cv2.imread(current_mini_mosaic_path)
        if image_color is None:
            print "WARNING: Could not read image {}. Skipping.".format(current_mini_mosaic_path)
            continue
        
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
        
        if result is None: 
            print "Initializing global mosaic with {}".format(current_mini_mosaic_path)
            result = image_color.copy() 
            result_gry = image_gray.copy()
            successful_stitches = 1
            try:
                cv2.imwrite(os.path.join(save_path, "global_mosaic_intermediate_{}.png".format(successful_stitches)), result)
                print "Saved initial global mosaic: {}".format(os.path.join(save_path, "global_mosaic_intermediate_{}.png".format(successful_stitches)))
            except Exception as e:
                print "ERROR writing initial intermediate mosaic: {}".format(e)
            continue # Skip to the next image if this was just the first one

        print "Attempting to stitch current global mosaic (W:{} x H:{}) with new mini-mosaic (W:{} x H:{})".format(
            result.shape[1], result.shape[0], image_color.shape[1], image_color.shape[0])
        
        h_time = time.time()
        # H_surf_result should be H_current_global_to_new_mini
        H_surf_result = surf(result_gry, image_gray) 
        elapsed_time_h = time.time() - h_time
        print "SURF homography estimation time: {:.2f}s".format(elapsed_time_h)

        # --- Robust Homography Check ---
        # Add more strict checks for homography validity
        if H_surf_result is None or \
           not isinstance(H_surf_result, np.ndarray) or \
           H_surf_result.shape != (3,3) or \
           np.all(H_surf_result == 0) or \
           not np.isfinite(H_surf_result).all(): # Check for NaN/Inf in the homography
            print "WARNING: SURF failed to find a valid homography for {}.".format(current_mini_mosaic_path)
            if H_surf_result is not None: print "Returned H:\n", H_surf_result
            print "Skipping this mini-mosaic due to invalid homography."
            
            H_flat_for_csv = ['failed_H_estimation'] * 9 
            status_for_csv = "failed_H_estimation"
        else:
            print "Homography Matrix found:\n", H_surf_result
            H_flat_for_csv = np.array(H_surf_result).flatten().astype(np.float64).tolist()
            status_for_csv = "success"
        
        # Log homography attempt (success or failure)
        try:
            with open(os.path.join(save_path, "H_surf_global_from_mini.csv"), 'a') as f1:
               wr = csv.writer(f1, delimiter=",", quoting = csv.QUOTE_NONE) # Removed escapechar as it defaults well
               wr.writerow(H_flat_for_csv)
            with open(os.path.join(save_path, "H_surf_global-from_mini_time_elapsed.csv"), 'a') as f2:
               twr = csv.writer(f2,  delimiter=",", quoting = csv.QUOTE_NONE) # Removed escapechar
               twr.writerow([elapsed_time_h, status_for_csv])
        except Exception as e:
            print "ERROR logging homography to CSV: {}".format(e)

        if status_for_csv != "success":
            continue # Skip mosaicking if H was invalid

        # --- Proceed with mosaicking if H is valid ---
        try:
            # image_color is img0 (new mini-mosaic)
            # result is img1 (current global mosaic)
            # H_surf_result is h_all (H_current_global_to_new_mini)
            result_after_stitch, _ = mosaicking(image_color, result, successful_stitches, H_surf_result, H_tp_dummy)
            
            # Check if mosaicking returned the original result (indicating an internal skip/error)
            # Use np.array_equal for content comparison since 'is' checks object identity
            if result_after_stitch is result or np.array_equal(result_after_stitch, result): 
                print "WARNING: Mosaicking function indicated a failure or skipped stitching for {}.".format(current_mini_mosaic_path)
            else:
                result = result_after_stitch
                result_gry = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                successful_stitches += 1
                print "Successfully stitched. Total successful stitches: {}".format(successful_stitches)
                # Save intermediate global mosaic after each successful stitch
                intermediate_save_path = os.path.join(save_path, "global_mosaic_intermediate_{}.png".format(successful_stitches))
                try:
                    cv2.imwrite(intermediate_save_path, result)
                    print "Saved intermediate global mosaic: {}".format(intermediate_save_path)
                except Exception as e:
                    print "ERROR writing intermediate global mosaic {}: {}".format(intermediate_save_path, e)

        except np.linalg.LinAlgError as lae:
            print "ERROR: LinAlgError during mosaicking for {}: {}. Skipping.".format(current_mini_mosaic_path, lae)
        except cv2.error as cve:
            print "ERROR: OpenCV error during mosaicking for {}: {}. Skipping.".format(current_mini_mosaic_path, cve)
        except Exception as e: # Catch any other unexpected errors
            print "ERROR: Unexpected error during mosaicking for {}: {}. Skipping.".format(current_mini_mosaic_path, e)
            import traceback
            traceback.print_exc(file=sys.stderr)

    # --- Final Save ---
    if result is not None and successful_stitches > 0 :
        final_save_path = os.path.join(save_path, "final_global_mosaic_{}_stitched_imgs.png".format(successful_stitches))
        try:
            cv2.imwrite(final_save_path, result)
            print "Final global mosaic saved to: {}".format(final_save_path)
        except cv2.error as e:
            print "ERROR saving final mosaic {}: {}".format(final_save_path, e)
        except Exception as e:
            print "ERROR: An unexpected error occurred while saving final mosaic {}: {}".format(final_save_path, e)
    elif result is not None and successful_stitches == 1:
        print "INFO: Only the first mini-mosaic was processed. Saved as intermediate_1 (if successful)."
    else:
        print "INFO: No images were successfully stitched into a global mosaic."
    
    print "Global assembly DONE!"
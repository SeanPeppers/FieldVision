'''
project: MaiZaic CUURENT VERSSIOOOONNNN

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
import cv2 as cv
import csv
import numpy as np
from numpy.linalg import inv, det, cond
from code.asift.asift import my_asift # Assuming my_asift is robust
from datetime import datetime
import time # ADDED: Import the time module
from math import radians, cos, sin, sqrt, atan2 # Corrected: Ensure individual math functions are imported
import sys
import logging
import rasterio
from rasterio.transform import rowcol, from_origin, xy
import exifread
import subprocess


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# GPS utility functions (copied for self-containment, similar to asift_homography_estimation.py)
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
                logger.debug(f"DEBUG(GPS_Extract): Found GPS in {os.path.basename(image_path)}: Lat={lat}, Lon={lon}")
            else:
                logger.debug(f"DEBUG(GPS_Extract): No GPS EXIF tags found in {os.path.basename(image_path)}.")
    except Exception as e:
        logger.warning(f"WARNING(GPS_Extract): Error extracting GPS from {os.path.basename(image_path)}: {e}")
    return gps

# Function to load raster transform from GeoTIFF (similar to stitcher.py)
def load_raster_transform(image_path):
    if image_path.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(image_path) as src:
                transform = src.transform
            return transform
        except Exception as e:
            logger.warning(f"WARNING: Error loading transform for {image_path}: {e}")
            return None
    else:
        logger.debug(f"DEBUG: Skipping transform loading for non-tiff file: {image_path}")
        return None

# Removed calculate_gps_homography as it's no longer used for pre-alignment.


def mosaicking(img0: np.ndarray, current_global_mosaic: np.ndarray, H_current_to_global: np.ndarray):
    """
    Stitches a new image (img0, the current mini-mosaic) onto the existing global mosaic canvas.

    Args:
        img0 (np.ndarray): The new image (current mini-mosaic frame) to be added to the global mosaic.
        current_global_mosaic (np.ndarray): The existing global mosaic image (previous state).
        H_current_to_global (np.ndarray): The homography matrix that transforms points from the
                                         current img0's coordinate space directly to the global mosaic's
                                         coordinate space.

    Returns:
        np.ndarray: The updated global mosaic.
    """
    logger.info("Adding new frame for mosaicking.")
    
    # This check ensures H_current_to_global is a valid 3x3 matrix before inversion.
    if H_current_to_global is None or np.isnan(H_current_to_global).any() or H_current_to_global.shape != (3, 3):
        logger.error(f"Invalid homography received for mosaicking. H is None, NaN, or incorrect shape ({H_current_to_global.shape if H_current_to_global is not None else 'None'}). Exiting.")
        sys.exit(1)

    h0, w0, _ = img0.shape # Dimensions of the current mini-mosaic
    h_prev_mosaic, w_prev_mosaic, _ = current_global_mosaic.shape # Dimensions of the previous global mosaic

    # Define corners of the current mini-mosaic (img0) in its own coordinate system
    corners0 = np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]], dtype=np.float32).reshape((-1, 1, 2))

    # Define corners of the previous global mosaic in its own coordinate system (relative to its top-left)
    corners_prev_mosaic = np.array([[0, 0], [w_prev_mosaic, 0], [w_prev_mosaic, h_prev_mosaic], [w_prev_mosaic, h_prev_mosaic]], dtype=np.float32).reshape((-1, 1, 2)) # FIX: Corner point was [img1.shape[1], 0]

    # Project the corners of the current mini-mosaic into the global coordinate system
    projected_corners0 = cv.perspectiveTransform(corners0, H_current_to_global)

    # All points defining the overall mosaic extent: previous mosaic corners + projected new mosaic corners
    all_corners = np.concatenate((corners_prev_mosaic, projected_corners0), axis=0) # FIX: Use all_corners here, not points1, points2

    # Get the min/max coordinates to determine the new canvas size
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Check for invalid canvas dimensions - Now exits on problematic size
    if x_max - x_min <= 0 or y_max - y_min <= 0:
        logger.error(f"Calculated mosaic dimensions are invalid: ({x_max - x_min}x{y_max - y_min}). Exiting.")
        sys.exit(1)

    # Create a translation matrix to shift all coordinates so that min_x and min_y become 0
    # This matrix effectively shifts the entire stitched scene to start at (0,0) of the new canvas.
    H_translation = np.array([[1, 0, -x_min], 
                              [0, 1, -y_min], 
                              [0, 0, 1]], dtype=np.float32)

    # Create the new larger canvas
    new_canvas_width = x_max - x_min
    new_canvas_height = y_max - y_min
    output_img = np.zeros((new_canvas_height, new_canvas_width, 3), dtype=np.uint8)

    # Apply the translation to the previous global mosaic and place it on the new canvas.
    warped_prev_mosaic = cv.warpPerspective(current_global_mosaic, H_translation, (new_canvas_width, new_canvas_height))
    
    # Warp the current mini-mosaic (img0) directly into the new global canvas.
    warped_current_mini_mosaic = cv.warpPerspective(img0, H_translation.dot(H_current_to_global), (new_canvas_width, new_canvas_height))
    
    # Blending (hard mask based)
    mask_new = (warped_current_mini_mosaic > 0).astype(np.uint8) * 255
    mask_new_eroded = cv.erode(mask_new, np.ones((10,10), np.uint8)) # Erode to avoid jagged edges

    temp_mosaic = cv.add(warped_prev_mosaic, 0, mask=cv.bitwise_not(mask_new_eroded))
    combined_mosaic = cv.add(temp_mosaic, warped_current_mini_mosaic, mask=mask_new_eroded)

    return combined_mosaic


def save_tiles(mosaic_image: np.ndarray, output_dir: str, tile_size: int, base_filename: str):
    """
    Saves a large mosaic image as a set of smaller tiles.
    """
    if tile_size <= 0:
        logger.warning("Tile size is 0 or less. Skipping tiling.")
        return

    h, w, _ = mosaic_image.shape
    tiles_output_dir = os.path.join(output_dir, f"{base_filename}_tiles_{tile_size}px")
    if not os.path.exists(tiles_output_dir):
        os.makedirs(tiles_output_dir)
        logger.info(f"Created tiling output directory: {tiles_output_dir}")

    num_cols = (w + tile_size - 1) // tile_size
    num_rows = (h + tile_size - 1) // tile_size

    logger.info(f"Tiling mosaic into {num_rows}x{num_cols} tiles of size {tile_size}x{tile_size}.")

    for i in range(num_rows):
        for j in range(num_cols):
            y_start = i * tile_size
            x_start = j * tile_size
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            tile = mosaic_image[y_start:y_end, x_start:x_end]

            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile_to_save = padded_tile
            else:
                tile_to_save = tile

            tile_filename = os.path.join(tiles_output_dir, f"{base_filename}_tile_{i:04d}_{j:04d}.tif")
            try:
                cv.imwrite(tile_filename, tile_to_save)
                logger.debug(f"Saved tile {i},{j} to {tile_filename}")
            except Exception as e:
                logger.error(f"Failed to save tile {tile_filename}: {e}")
    logger.info(f"Finished saving {num_rows * num_cols} tiles to {tiles_output_dir}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories (mini-mosaics)")
    parser.add_argument('-save_path', dest='save_path', default="global_mosaic/", type=str, help="path to save result")
    parser.add_argument('-tile_size_px', type=int, default=256, 
                        help="Size of square tiles (e.g., 256, 512). If 0, no tiling is performed. Default is 256 to enable tiling by default.")
    parser.add_argument('-asift_scale', type=float, default=0.7, # Scale for ASIFT processing
                        help="Scale factor for images fed to ASIFT feature detection (e.g., 0.5 to halve dimensions). Reduces memory/time for ASIFT. Default is 0.7.")
    # Removed pixel_res_m_per_px argument as it's no longer used for pre-alignment.
    
    args = parser.parse_args()

    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created save path directory: {save_path}")

    global_mosaic = None # Stores the accumulating full-scale mosaic
    global_mosaic_gry = None # Stores the grayscale of the accumulating full-scale mosaic
    H_global = np.identity(3, dtype=np.float32) # Accumulates homographies from first mini-mosaic to current global mosaic

    image_index = -1
    counter = 0
    
    # Process image paths to flatten directories into a single list of files
    all_image_files = []
    supported_extensions = [".jpeg", ".jpg", ".png", ".tif", ".tiff"] 
    for path_arg in args.image_path: # Use args.image_path here
        if os.path.isdir(path_arg):
            logger.info(f"Scanning directory for images: {path_arg}")
            for file_path in sorted(os.listdir(path_arg)):
                if os.path.splitext(file_path)[1].lower() in supported_extensions:
                    all_image_files.append(os.path.join(path_arg, file_path)) 
        elif os.path.isfile(path_arg) and os.path.splitext(path_arg)[1].lower() in supported_extensions:
            all_image_files.append(path_arg)
        else:
            logger.warning(f"Path '{path_arg}' does not exist or is not a supported image file/directory. Skipping.")

    if not all_image_files:
        logger.error("No valid image files found to process. Exiting.")
        sys.exit(1)

    logger.info(f"Found {len(all_image_files)} image files (mini-mosaics) for global mosaicking.")

    # Define the scaling matrix for ASIFT and for scaling back the homography
    asift_scale_factor = args.asift_scale
    scale_matrix_asift = np.array([[asift_scale_factor, 0, 0], 
                                   [0, asift_scale_factor, 0], 
                                   [0, 0, 1]], dtype=np.float32)
    if asift_scale_factor == 0:
        logger.error("ASIFT scale factor cannot be zero. Exiting.")
        sys.exit(1)
    scale_matrix_asift_inv = np.array([[1.0/asift_scale_factor, 0, 0], 
                                       [0, 1.0/asift_scale_factor, 0], 
                                       [0, 0, 1]], dtype=np.float32)

    # Removed GPS tracking variables for pre-alignment.
    # prev_mini_mosaic_gps = None
    # current_global_mosaic_transform = None 
    
    for image_path in all_image_files:
        logger.info(f"Reading frame: {image_path}")
        
        # Load the current mini-mosaic image (which should be a GeoTIFF from stitcher.py)
        try:
            with rasterio.open(image_path) as src:
                # Read all bands and transpose for OpenCV (HxWxB)
                image_bands = src.read()
                if src.count == 3:
                    image_color_original_res = np.transpose(image_bands, (1, 2, 0))
                elif src.count == 1:
                    image_color_original_res = cv.cvtColor(image_bands[0], cv.COLOR_GRAY2BGR)
                else:
                    logger.error(f"Unsupported band count {src.count} in {image_path}. Skipping.")
                    continue
                image_color_original_res = cv.cvtColor(image_color_original_res, cv.COLOR_RGB2BGR) # Convert to BGR
                
                # We still read these, even if not used for pre-alignment, in case tiling needs them (future use)
                # or for manual verification of input GeoTIFFs.
                current_mini_mosaic_transform = src.transform 
                current_mini_mosaic_gps = xy(current_mini_mosaic_transform, 0, 0) 
                current_mini_mosaic_gps = (current_mini_mosaic_gps[1], current_mini_mosaic_gps[0])
                
        except Exception as e:
            logger.error(f"Failed to load image {image_path} using rasterio (or get GeoTIFF data). Falling back to cv.imread and EXIF GPS: {e}")
            image_color_original_res = cv.imread(image_path) # Fallback to OpenCV's imread
            if image_color_original_res is None:
                logger.error(f"Failed to load image {image_path} with cv.imread as well. Skipping this frame.")
                continue
            
            # If not a GeoTIFF, try to extract EXIF GPS (less precise for a mosaic)
            current_mini_mosaic_gps = extract_gps_from_image(image_path)
            current_mini_mosaic_transform = None 
            logger.warning(f"WARNING: No GeoTIFF transform for {os.path.basename(image_path)}. Using EXIF GPS fallback: {current_mini_mosaic_gps}")


        filename = os.path.basename(image_path)
        
        image_index += 1

        if image_index == 0:
            logger.info(f"Initializing global mosaic with first mini-mosaic (index {image_index}).")
            global_mosaic = image_color_original_res # 'global_mosaic' is the accumulating full-scale mosaic
            global_mosaic_gry = cv.cvtColor(global_mosaic, cv.COLOR_BGR2GRAY) # Grayscale of full-scale mosaic
            
            # For logging/future use, store info of the first mini-mosaic
            # prev_mini_mosaic_gps = current_mini_mosaic_gps # Not needed for pre-alignment
            # current_global_mosaic_transform = current_mini_mosaic_transform # Not needed for pre-alignment
            
            continue

        logger.info(f"Processing frame {image_index}, counter {counter}.")

        # --- Prepare images for ASIFT at the specified asift_scale ---
        current_img_for_asift_w = int(image_color_original_res.shape[1] * asift_scale_factor)
        current_img_for_asift_h = int(image_color_original_res.shape[0] * asift_scale_factor)
        current_img_for_asift_w = max(1, current_img_for_asift_w)
        current_img_for_asift_h = max(1, current_img_for_asift_h)
        
        image_color_for_asift = image_color_original_res
        if asift_scale_factor != 1.0:
            image_color_for_asift = cv.resize(image_color_original_res, (current_img_for_asift_w, current_img_for_asift_h))
        image_gray_for_asift = cv.cvtColor(image_color_for_asift, cv.COLOR_BGR2GRAY) 

        mosaic_for_asift_w = int(global_mosaic.shape[1] * asift_scale_factor)
        mosaic_for_asift_h = int(global_mosaic.shape[0] * asift_scale_factor)
        mosaic_for_asift_w = max(1, mosaic_for_asift_w)
        mosaic_for_asift_h = max(1, mosaic_for_asift_h)

        global_mosaic_gry_for_asift = global_mosaic_gry
        if asift_scale_factor != 1.0:
            global_mosaic_gry_for_asift = cv.resize(global_mosaic_gry, (mosaic_for_asift_w, mosaic_for_asift_h))

        # Removed GPS-based initial homography and pre-alignment.
        # H_gps_initial_guess = calculate_gps_homography(...)
        # image_gray_pre_aligned_by_gps = cv.warpPerspective(...) or image_gray_for_asift

        logger.info(f"Running my_asift on scaled images (scale: {asift_scale_factor:.2f})...")
        h_time = time.time()
        # ASIFT now directly operates on the scaled grayscale images, without GPS pre-alignment.
        # H_computed_scaled maps global_mosaic_gry_for_asift to image_gray_for_asift
        H_computed_scaled = my_asift(global_mosaic_gry_for_asift, image_gray_for_asift) 
        elapsed_time_h = time.time()-h_time

        # --- Check H for None/NaN/Shape from ASIFT ---
        if H_computed_scaled is None or np.isnan(H_computed_scaled).any() or H_computed_scaled.shape != (3, 3):
            logger.error(f"my_asift returned None or invalid H for frame {filename}. Exiting to prevent crash or corrupted output.")
            sys.exit(1)

        # Scale the homography back to the original full resolution
        H_final_for_mosaicking = np.dot(np.dot(scale_matrix_asift_inv, H_computed_scaled), scale_matrix_asift)
        # Removed homography combination here as GPS pre-alignment is not used.

        H_flat = np.array(H_final_for_mosaicking).flatten().astype(np.float64) 
        logger.info(f"Final Homography for mosaicking (full scale, flattened): {H_flat}")
        
        homography_csv_path = os.path.join(save_path, "H_asift_global.csv") 
        time_elapsed_csv_path = os.path.join(save_path, "H_asift_time_elapsed.csv")

        try:
            with open(homography_csv_path, 'a', newline='') as f1:
               wr = csv.writer(f1, delimiter=",", quoting = csv.QUOTE_NONE)
               wr.writerow(H_flat)
            logger.info(f"Homography saved to {homography_csv_path}")
        except Exception as e:
            logger.error(f"Failed to write homography to CSV: {e}")

        try:
            with open(time_elapsed_csv_path, 'a', newline='') as f2:
               twr = csv.writer(f2,  delimiter=",", quoting = csv.QUOTE_NONE)
               twr.writerow([elapsed_time_h])
            logger.info(f"Homography computation time saved to {time_elapsed_csv_path}")
        except Exception as e:
            logger.error(f"Failed to write elapsed time to CSV: {e}")
         
        # Call mosaicking function with original resolution images and the new final homography
        global_mosaic = mosaicking(image_color_original_res, global_mosaic, H_final_for_mosaicking)

        counter+=1
        output_mosaic_filename = os.path.join(save_path, f"global_mosaic_{counter:03d}.tif") 
        try:
            cv.imwrite(output_mosaic_filename, global_mosaic) # Saves intermediate mosaics
            logger.info(f"Intermediate mosaic saved to {output_mosaic_filename}")
        except Exception as e:
            logger.error(f"Failed to save intermediate mosaic {output_mosaic_filename}: {e}")

        global_mosaic_gry = cv.cvtColor(global_mosaic, cv.COLOR_BGR2GRAY) # Update grayscale of growing full-scale mosaic
        
        # Removed global mosaic transform update logic as it was related to GPS pre-alignment.
        # current_global_mosaic_transform = ...
        # prev_mini_mosaic_gps = ...
    
    logger.info("Global Mosaicking DONE!") # Final log outside the loop

    # --- Final mosaic saving and tiling ---
    final_mosaic_filename_base = f"final_global_mosaic_{counter:03d}"
    
    # Save the full mosaic (as you currently do)
    final_mosaic_full_path = os.path.join(save_path, f"{final_mosaic_filename_base}.tif")
    try:
        cv.imwrite(final_mosaic_full_path, global_mosaic) # 'global_mosaic' is the final mosaic image
        logger.info(f"Final global mosaic saved to {final_mosaic_full_path}")
    except Exception as e:
        logger.error(f"Failed to save final mosaic {final_mosaic_full_path}: {e}")

    # Now, call the tiling function if tile_size_px is specified (default is 256, so it will run by default)
    if args.tile_size_px > 0:
        save_tiles(global_mosaic, save_path, args.tile_size_px, final_mosaic_filename_base)
    
    logger.info("Global Mosaicking process finished.")


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    elapsed = (datetime.now() - start_time).total_seconds()
    print('Total Global Mosaicking time elapsed: ', elapsed)
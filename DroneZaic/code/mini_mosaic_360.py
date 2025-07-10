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
import cv2 as cv
import csv
import numpy as np
from numpy.linalg import inv, det, cond
from code.asift.asift import my_asift
from datetime import datetime
import time
import math
import sys
import logging
import rasterio # Added rasterio import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mosaicking(img0: np.ndarray, img1: np.ndarray, counter: int, h_all: np.ndarray, H_tp: np.ndarray):
    """
    Stitches a new image (img0) onto an existing mosaic (img1) using a homography matrix.

    Args:
        img0 (np.ndarray): The new image (current frame) to be added to the mosaic.
        img1 (np.ndarray): The existing global mosaic (previous state).
        counter (int): The current frame counter, used for logging and error messages.
        h_all (np.ndarray): The 3x3 homography matrix (img1 -> img0).
        H_tp (np.ndarray): Placeholder, unused in this implementation.

    Returns:
        tuple: A tuple containing:
            - output_img (np.ndarray): The updated global mosaic.
            - H_translation (np.ndarray): The translation matrix applied to the mosaic.
                                         (Note: This return value is not used externally in the current main loop.)
    """
    logger.info("Adding new frame for mosaicking.")
    
    # This check ensures h_all is a valid 3x3 matrix before inversion.
    if h_all is None or np.isnan(h_all).any() or h_all.shape != (3, 3):
        logger.error(f"Invalid homography received for mosaicking frame {counter}. H is None, NaN, or incorrect shape ({h_all.shape if h_all is not None else 'None'}). Exiting.")
        sys.exit(1)

    # The homography `h_all` from `my_asift` is `prev_mosaic -> current_image`.
    # For `cv.warpPerspective` to warp `current_image` (`img0`) onto `prev_mosaic`'s canvas,
    # we need the inverse transformation: `current_image -> prev_mosaic`.
    h_all_inverted = inv(h_all) 
    
    points0 = np.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))

    # get the transformed corner from new image (img0) in the coordinate space of img1
    points2 = cv.perspectiveTransform(points0, h_all_inverted)

    # Check for wildly transformed corners - Now exits on extreme coordinates
    max_coord_val = max(img0.shape[0], img0.shape[1], img1.shape[0], img1.shape[1]) * 5
    if np.any(np.abs(points2) > max_coord_val) and counter > 0:
        logger.error(f"Transformed corners for frame {counter} ({points2}) are too extreme (>{max_coord_val}). This indicates a bad homography. Exiting to prevent OOM or corrupted output.")
        sys.exit(1)

    # get the max and min coordinate of mosaic images
    points = np.concatenate((points1, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)

    # Check for invalid canvas dimensions - Now exits on problematic size
    if x_max - x_min <= 0 or y_max - y_min <= 0:
        logger.error(f"Calculated mosaic dimensions are invalid: ({x_max - x_min}x{y_max - y_min}). Exiting.")
        sys.exit(1)

    # additional translation from offset
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    output_img = np.zeros(( y_max - y_min,x_max - x_min,3), dtype=np.uint8)
    # The next line places img1 (previous mosaic) onto the new canvas.
    # Its coordinates are adjusted by -y_min, -x_min to shift everything to positive.
    output_img[-y_min:img1.shape[0] - y_min, -x_min:img1.shape[1] - x_min] = img1
    
    # apply homography to new image (img0).
    # H_translation.dot(h_all_inverted) applies the homography first, then the translation.
    warped_img = cv.warpPerspective(img0, H_translation.dot(h_all_inverted),(x_max - x_min, y_max - y_min))
    
    # Blending (hard mask based)
    mask2 = (warped_img > 0).astype(np.uint8) * 255 
    mask3 = cv.erode(mask2, np.ones((10,10), np.uint8)) 
    masked_mosaic = cv.bitwise_and(output_img,  cv.bitwise_not(mask3))
    warped_img2 = cv.bitwise_and(warped_img, mask3)
    output_img = cv.bitwise_or(warped_img2,  masked_mosaic)

    return output_img, H_translation


def save_tiles(mosaic_image: np.ndarray, output_dir: str, tile_size: int, base_filename: str):
    """
    Saves a large mosaic image as a set of smaller tiles.

    Args:
        mosaic_image (np.ndarray): The full stitched mosaic image.
        output_dir (str): The base directory to save the tiles.
        tile_size (int): The size of each square tile (e.g., 256).
        base_filename (str): Base name for tiles (e.g., "final_mosaic").
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

            # Handle partial tiles: pad with black or specific color
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
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    # REVERTED: -hm is no longer required and is unused as homographies are computed here
    parser.add_argument('-hm', '--homography', type=str, help='(UNUSED) txt file that stores homography matrices')
    parser.add_argument('-save_path', dest='save_path', default="global_mosaic/", type=str, help="path to save result")
    parser.add_argument('-tile_size_px', type=int, default=256, 
                        help="Size of square tiles (e.g., 256, 512). If 0, no tiling is performed. Default is 256 to enable tiling by default.")
    # ASIFT scale is now active again in this script
    parser.add_argument('-asift_scale', type=float, default=0.5, # Reverted default to 0.5 for memory
                        help="Scale factor for images fed to ASIFT feature detection (e.g., 0.5 to halve dimensions). Reduces memory/time for ASIFT. Default is 0.5.")
    
    args = parser.parse_args()

    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created save path directory: {save_path}")

    result = None # Stores the accumulating full-scale mosaic
    result_gry = None # Stores the grayscale of the accumulating full-scale mosaic

    image_paths = args.image_path # Original image_paths from CLI
    homography_file_path = args.homography # This is unused now
    image_index = -1
    counter = 0
    H_tp = np.array([[0,0,0],[0,0,0],[0,0,0]]) 
    
    # Process image paths to flatten directories into a single list of files
    all_image_files = []
    supported_extensions = [".jpeg", ".jpg", ".png", ".tif", ".tiff"] # Added more extensions for robustness
    for path_arg in image_paths:
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

    logger.info(f"Found {len(all_image_files)} image files for mosaicking.")

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


    for image_path in all_image_files:
        logger.info(f"Reading frame: {image_path}")
        
        # Using rasterio to load images more robustly/efficiently, especially if they are TIFFs
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
        except Exception as e:
            logger.error(f"Failed to load image {image_path} using rasterio. Falling back to cv.imread: {e}")
            image_color_original_res = cv.imread(image_path) # Fallback to OpenCV's imread
            if image_color_original_res is None:
                logger.error(f"Failed to load image {image_path} with cv.imread as well. Skipping this frame.")
                continue


        filename = os.path.basename(image_path)
        
        image_index += 1

        if image_index == 0:
            logger.info(f"Initializing mosaic with first image (index {image_index}).")
            result = image_color_original_res # 'result' is the accumulating full-scale mosaic
            result_gry = cv.cvtColor(result, cv.COLOR_BGR2GRAY) # Grayscale of full-scale mosaic
            continue

        logger.info(f"Processing frame {image_index}, counter {counter}.")

        # --- Prepare images for ASIFT at the specified asift_scale ---
        # Scale down the current image (new frame) for ASIFT processing
        current_img_for_asift_w = int(image_color_original_res.shape[1] * asift_scale_factor)
        current_img_for_asift_h = int(image_color_original_res.shape[0] * asift_scale_factor)
        current_img_for_asift_w = max(1, current_img_for_asift_w) # Ensure minimum 1x1 dimensions
        current_img_for_asift_h = max(1, current_img_for_asift_h)
        
        image_color_for_asift = image_color_original_res
        if asift_scale_factor != 1.0:
            image_color_for_asift = cv.resize(image_color_original_res, (current_img_for_asift_w, current_img_for_asift_h))
        image_gray_for_asift = cv.cvtColor(image_color_for_asift, cv.COLOR_BGR2GRAY) # Use BGR2GRAY for consistent loaded color image

        # Scale down the current 'result' mosaic for ASIFT processing
        mosaic_for_asift_w = int(result.shape[1] * asift_scale_factor)
        mosaic_for_asift_h = int(result.shape[0] * asift_scale_factor)
        mosaic_for_asift_w = max(1, mosaic_for_asift_w)
        mosaic_for_asift_h = max(1, mosaic_for_asift_h)

        result_gry_for_asift = result_gry
        if asift_scale_factor != 1.0:
            result_gry_for_asift = cv.resize(result_gry, (mosaic_for_asift_w, mosaic_for_asift_h))

        logger.info(f"Running my_asift on scaled images (scale: {asift_scale_factor:.2f})...")
        h_time = time.time()
        # my_asift calculates H from result_gry_for_asift (scaled previous mosaic) to image_gray_for_asift (scaled current image)
        H_computed_scaled = my_asift(result_gry_for_asift, image_gray_for_asift) 
        elapsed_time_h = time.time()-h_time

        # --- Check H for None/NaN/Shape and upscale it if valid ---
        if H_computed_scaled is None or np.isnan(H_computed_scaled).any() or H_computed_scaled.shape != (3, 3):
            logger.error(f"my_asift returned None or invalid H for frame {filename}. Exiting to prevent crash or corrupted output.")
            sys.exit(1) # Exits if H is problematic

        # Scale the homography back to the original full resolution
        H_computed_full_scale = np.dot(np.dot(scale_matrix_asift_inv, H_computed_scaled), scale_matrix_asift)
        H_computed = H_computed_full_scale # Use this full-scale homography for mosaicking

        H_flat = np.array(H_computed).flatten().astype(np.float64)
        logger.info(f"Computed Homography (full scale, flattened): {H_flat}")
        
        homography_csv_path = os.path.join(save_path, "H_asift.csv")
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
         
        # Call mosaicking function with original resolution images and full-scale homography
        # img0: current image at original resolution
        # img1: accumulating mosaic at original resolution
        # h_all: H_computed (full-scale homography)
        result, H_tp = mosaicking(image_color_original_res, result, counter, H_computed, H_tp) 

        counter+=1
        output_mosaic_filename = os.path.join(save_path, f"global_mosaic_{counter:03d}.tif") # Format with leading zeros
        try:
            cv.imwrite(output_mosaic_filename, result) # Saves intermediate mosaics
            logger.info(f"Intermediate mosaic saved to {output_mosaic_filename}")
        except Exception as e:
            logger.error(f"Failed to save intermediate mosaic {output_mosaic_filename}: {e}")

        result_gry = cv.cvtColor(result, cv.COLOR_BGR2GRAY) # Update grayscale of growing full-scale mosaic

    # --- Final mosaic saving and tiling ---
    final_mosaic_filename_base = f"final_global_mosaic_{counter:03d}"
    
    # Save the full mosaic (as you currently do)
    final_mosaic_full_path = os.path.join(save_path, f"{final_mosaic_filename_base}.tif")
    try:
        cv.imwrite(final_mosaic_full_path, result) # 'result' is the final mosaic image
        logger.info(f"Final global mosaic saved to {final_mosaic_full_path}")
    except Exception as e:
        logger.error(f"Failed to save final mosaic {final_mosaic_full_path}: {e}")

    # Now, call the tiling function if tile_size_px is specified (default is 256, so it will run by default)
    if args.tile_size_px > 0:
        save_tiles(result, save_path, args.tile_size_px, final_mosaic_filename_base)
    
    logger.info("Mosaicking DONE!")
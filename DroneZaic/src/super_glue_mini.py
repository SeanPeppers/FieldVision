import torch
import cv2
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor
import matplotlib.pyplot as plt
import gc # Import garbage collector
import os # Import os module for path manipulation

# Ensure you have a GPU for faster processing if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load SuperPoint and SuperGlue models
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 2048 # Limit keypoints to reduce memory and computation
    },
    'superglue': {
        'weights': 'outdoor', # 'indoor' or 'outdoor'
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2
    }
}
matching = Matching(config).eval().to(device)

# --- Memory Optimization Parameters ---
DOWNSCALE_FACTOR = 0.25 # Process at 25% of original resolution

# Function to prepare image for SuperGlue (grayscale, normalized tensor, potentially downscaled)
def prepare_image_for_superglue(image_cv2, device, downscale_factor=1.0):
    if downscale_factor != 1.0:
        h, w = image_cv2.shape[:2]
        new_h, new_w = int(h * downscale_factor), int(w * downscale_factor)
        image_resized = cv2.resize(image_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image_resized = image_cv2

    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    return frame2tensor(image_gray, device)

# --- Docker-specific Path Configuration ---
# Base directory containing all asift_mini_partition/group_XXX directories
BASE_INPUT_DIR = "/app/outputs/asift_mini_partition"
GLOBAL_OUTPUT_DIR = "/app/outputs/glue_mini_mosaics"

# Create global output directory if it doesn't exist
os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)
print(f"Base input directory: {BASE_INPUT_DIR}")
print(f"Global output directory: {GLOBAL_OUTPUT_DIR}")

# Get all partition directories (e.g., group_001, group_002)
partition_dirs = sorted([d for d in os.listdir(BASE_INPUT_DIR) if os.path.isdir(os.path.join(BASE_INPUT_DIR, d)) and d.startswith('group_')])

if not partition_dirs:
    raise ValueError(f"No 'group_XXX' directories found in {BASE_INPUT_DIR}. Please check your input structure.")

print(f"Found partition directories: {partition_dirs}")

# Loop through each partition directory
for partition_name in partition_dirs:
    INPUT_IMAGE_DIR = os.path.join(BASE_INPUT_DIR, partition_name)
    # Create a specific output directory for each partition's mosaic
    PARTITION_OUTPUT_DIR = os.path.join(GLOBAL_OUTPUT_DIR, partition_name)
    os.makedirs(PARTITION_OUTPUT_DIR, exist_ok=True)

    print(f"\n========================================================")
    print(f"--- Processing partition: {partition_name} ---")
    print(f"Input directory for this partition: {INPUT_IMAGE_DIR}")
    print(f"Output directory for this partition: {PARTITION_OUTPUT_DIR}")
    print(f"========================================================")

    # --- Multi-Image Stitching Logic (re-initialize for each partition) ---
    # Load images from the current partition directory
    image_filenames = sorted([f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))])

    # Read images into a dictionary
    images = {}
    for filename in image_filenames:
        filepath = os.path.join(INPUT_IMAGE_DIR, filename)
        img = cv2.imread(filepath)
        if img is not None:
            images[filename] = img
        else:
            print(f"Warning: Could not load image {filepath}. Skipping.")

    image_filenames = sorted(list(images.keys())) # Update filenames list to only include loaded images

    print(f"Images identified for stitching in {partition_name} (sorted): {image_filenames}")

    if len(image_filenames) < 2:
        print(f"Skipping partition {partition_name}: Not enough images (need at least 2) for stitching. Found {len(image_filenames)}.")
        continue # Skip to the next partition

    # Start with the first image as the base panorama. Use .copy() to prevent accidental modifications
    stitched_panorama = images[image_filenames[0]].copy()

    # Loop through the rest of the images, stitching them one by one
    for i in range(1, len(image_filenames)):
        current_image_name = image_filenames[i]
        current_image = images[current_image_name].copy()

        print(f"\n--- Stitching step {i}/{len(image_filenames) - 1} for {partition_name} ---")
        print(f"Stitching {current_image_name} (original size: {current_image.shape[1]}x{current_image.shape[0]}) to the current panorama...")

        # Prepare images for SuperGlue (downscaled for feature extraction)
        img0_torch_downscaled = prepare_image_for_superglue(stitched_panorama, device, DOWNSCALE_FACTOR)
        img1_torch_downscaled = prepare_image_for_superglue(current_image, device, DOWNSCALE_FACTOR)

        # Perform feature extraction and matching
        with torch.no_grad():
            pred = matching({'image0': img0_torch_downscaled, 'image1': img1_torch_downscaled})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0 = pred['keypoints0']
        kpts1 = pred['keypoints1']
        matches = pred['matches0']

        # Clean up PyTorch tensors no longer needed to free GPU memory
        del img0_torch_downscaled, img1_torch_downscaled, pred
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        print(f"Found {len(mkpts0)} matches between panorama and {current_image_name}.")

        if len(mkpts0) > 4: # Need at least 4 points for homography
            # Scale keypoints back to original image dimensions for homography calculation
            mkpts0_original_scale = mkpts0 / DOWNSCALE_FACTOR
            mkpts1_original_scale = mkpts1 / DOWNSCALE_FACTOR

            H, mask = cv2.findHomography(mkpts1_original_scale, mkpts0_original_scale, cv2.RANSAC, 5.0)

            if H is None:
                print(f"Homography estimation failed for {current_image_name} in {partition_name}. Skipping this image.")
                continue # Skip to the next image

            # Calculate the size of the new panorama canvas
            h0, w0 = stitched_panorama.shape[:2]
            h1, w1 = current_image.shape[:2]

            corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
            transformed_corners1 = cv2.perspectiveTransform(corners1, H)

            all_corners = np.vstack((np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2), transformed_corners1))

            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

            t_x = -xmin
            t_y = -ymin
            translation_mat = np.array([[1, 0, t_x],
                                        [0, 1, t_y],
                                        [0, 0, 1]])

            H_translated = translation_mat @ H

            # Create a new blank canvas for the combined panorama
            new_width = xmax - xmin
            new_height = ymax - ymin

            # Ensure dimensions are positive after transformation
            if new_width <= 0 or new_height <= 0:
                print(f"Invalid panorama dimensions ({new_width}x{new_height}) after stitching {current_image_name} in {partition_name}. Skipping.")
                continue

            new_panorama_canvas = np.zeros((new_height, new_width, 3), dtype=stitched_panorama.dtype)

            # Place the transformed current_image onto the new canvas
            warped_current_image = cv2.warpPerspective(current_image, H_translated, (new_width, new_height))

            # Place the existing stitched_panorama onto the new canvas, shifted by the translation
            mask_old_pano = np.zeros((new_height, new_width), dtype=np.uint8)
            mask_old_pano[t_y:t_y + h0, t_x:t_x + w0] = 255

            new_panorama_canvas[mask_old_pano == 255] = stitched_panorama[mask_old_pano[t_y:t_y + h0, t_x:t_x + w0] == 255]

            # Blend the warped_current_image with the existing content on the new_panorama_canvas
            gray_warped = cv2.cvtColor(warped_current_image, cv2.COLOR_BGR2GRAY)
            ret, mask_warped = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

            mask_warped_inv = cv2.bitwise_not(mask_warped)

            new_panorama_canvas_masked_old = cv2.bitwise_and(new_panorama_canvas, new_panorama_canvas, mask=mask_warped_inv)

            new_panorama_canvas = cv2.add(new_panorama_canvas_masked_old, warped_current_image)

            # Update the main panorama with the newly stitched result
            stitched_panorama = new_panorama_canvas

            # Clean up variables from current iteration
            del kpts0, kpts1, matches, valid, mkpts0, mkpts1, mkpts0_original_scale, mkpts1_original_scale, H, mask, corners1, transformed_corners1, all_corners, translation_mat, H_translated, warped_current_image, new_panorama_canvas_masked_old, mask_warped, mask_warped_inv, gray_warped, mask_old_pano
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        else:
            print(f"Not enough matches found to stitch {current_image_name} in {partition_name}. Skipping this image.")

    # Display and save the final stitched panorama for the current partition
    print(f"\n--- Final Display for {partition_name} ---")
    print(f"Attempting to display the final stitched panorama, which has dimensions: {stitched_panorama.shape[1]}x{stitched_panorama.shape[0]}")

    if stitched_panorama is not None and stitched_panorama.shape[0] > 0 and stitched_panorama.shape[1] > 0:
        # Displaying with plt.show() might not be ideal in a headless Docker environment
        # You'll mostly rely on the saved image.
        # plt.figure(figsize=(20, 15))
        # plt.imshow(cv2.cvtColor(stitched_panorama, cv2.COLOR_BGR2RGB))
        # plt.title(f"Final Stitched Panorama ({len(image_filenames)} Images) - SuperGlue - {partition_name}")
        # plt.axis('off')
        # plt.show() # This will block execution in some environments

        output_filename = os.path.join(PARTITION_OUTPUT_DIR, f"final_stitched_superglue_panorama_{partition_name}.jpg")
        try:
            cv2.imwrite(output_filename, stitched_panorama)
            print(f"Final stitched panorama for {partition_name} saved as {output_filename}")
        except Exception as e:
            print(f"Error saving stitched image for {partition_name}: {e}")
    else:
        print(f"Stitching failed or the final panorama for {partition_name} is empty. No image to display or save.")

    # Clear variables for the next partition to free up memory
    del stitched_panorama, images, image_filenames
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

print("\n--- All partitions processed. ---")
import argparse
import numpy as np
import cv2
import glob
import os
import sys # Added for sys.exit()

'''
example syntax

python calibration.py -image_path /path/to/input_frames_directory/ -save_path /path/to/output_directory/

Note: Ensure -image_path points to the directory containing the .png files,
      and it should usually end with a '/'.
'''

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('-image_path', type=str,  help="Path to the directory containing raw .png images.")
parser.add_argument('-save_path', type=str,  help="Path to the base directory where calibrated images will be saved (in a 'calibrated' subfolder).")
parser.add_argument('-xxx', '--xxx', dest='xxx', default = 0, type=float,  help='x rad calib for gimbal')
parser.add_argument('-yyy', '--yyy', dest='yyy', default = 0, type=float,  help='y rad calib for gimbal')
parser.add_argument('-zzz', '--zzz', dest='zzz', default = 0, type=float,  help='z rad calib for gimbal')

args = parser.parse_args()

# --- Input Validation and Path Setup ---
if not args.image_path:
    print "ERROR: -image_path argument is required."
    sys.exit(1)
if not args.save_path:
    print "ERROR: -save_path argument is required."
    sys.exit(1)

if not os.path.isdir(args.image_path):
    print "ERROR: Provided -image_path is not a valid directory: {}".format(args.image_path)
    sys.exit(1)

# Ensure image_path ends with a separator for consistent glob and path stripping
image_search_path = os.path.join(args.image_path, '') # Adds trailing slash if missing
glob_pattern = os.path.join(image_search_path, '*.png')
calibrated_output_dir = os.path.join(args.save_path, 'calibrated')

print "INFO: Input image path pattern: {}".format(glob_pattern)
print "INFO: Calibrated images will be saved to: {}".format(calibrated_output_dir)

images = sorted(glob.glob(glob_pattern))

print "INFO: Found {} images to calibrate.".format(len(images))

if not images:
    print "WARNING: No images found matching the pattern '{}'. Exiting.".format(glob_pattern)
    sys.exit(0)

if not os.path.exists(calibrated_output_dir):
   try:
       os.makedirs(calibrated_output_dir)
       print "INFO: Created output directory: {}".format(calibrated_output_dir)
   except OSError as e:
       print "ERROR: Could not create output directory {}: {}".format(calibrated_output_dir, e)
       sys.exit(1)

counter = 0

# --- Gimbal calibration functions ---
def roty(theta):
  ct = np.cos(theta)
  st = np.sin(theta)
  R_matrix=np.array([[ct, 0, st],[0,1,0],[-st, 0, ct]])
  return R_matrix

def rotx(t):
  ct = np.cos(t)
  st = np.sin(t)
  R_matrix=np.array([[1,0,0],[0, ct, -st],[0, st, ct]])
  return R_matrix

def rotz(t):
  ct = np.cos(t)
  st = np.sin(t)
  R_matrix=np.array([[ct,-st,0],[st, ct, 0],[0, 0, 1]])
  return R_matrix

print "INFO: Initializing calibration parameters..." # Changed from 'test'

R_calib = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float) # Default to identity, ensure float

if args.xxx != 0 and args.zzz == 0 and args.yyy == 0:
    R_calib = rotx(args.xxx)
    print "INFO: Applying X-axis rotation: {} radians".format(args.xxx)
elif args.yyy != 0 and args.zzz == 0 and args.xxx == 0:
    R_calib = roty(args.yyy)
    print "INFO: Applying Y-axis rotation: {} radians".format(args.yyy)
elif args.zzz != 0 and args.xxx == 0 and args.yyy == 0:
    R_calib = rotz(args.zzz)
    print "INFO: Applying Z-axis rotation: {} radians".format(args.zzz)
elif args.xxx != 0 and args.yyy != 0 and args.zzz == 0 :
    R_calib = np.dot(roty(args.yyy),rotx(args.xxx))
    print "INFO: Applying Y then X rotation: Y={}, X={}".format(args.yyy, args.xxx)
elif args.xxx != 0 and args.zzz != 0 and args.yyy == 0:
    R_calib = np.dot(rotz(args.zzz),rotx(args.xxx))
    print "INFO: Applying X then Z rotation: X={}, Z={}".format(args.xxx, args.zzz)
elif args.zzz != 0 and args.yyy != 0 and args.xxx == 0:
    R_calib = np.dot(roty(args.yyy),rotz(args.zzz))
    print "INFO: Applying Z then Y rotation: Z={}, Y={}".format(args.zzz, args.yyy)
elif args.xxx != 0 and args.yyy != 0 and args.zzz != 0 :
    print "INFO: Applying 3-axis rotation (Y then X then Z): Y={}, X={}, Z={}".format(args.yyy, args.xxx, args.zzz)
    R_calib = np.dot(np.dot(roty(args.yyy),rotx(args.xxx)), rotz(args.zzz))
else:
    print "INFO: No gimbal rotation arguments provided. Using identity matrix for R_calib."

# --- Camera and Calibration Parameters ---
K_intrinsic = np.array([[2359.79036,0, 2031.0],[ 0, 2359.30434, 1046.5],[ 0 ,  0,   1.0]])

# Homography for gimbal effect correction (projects points from original K frame to R_calib rotated K frame)
H1_gimbal = np.dot(np.dot(K_intrinsic, R_calib), np.linalg.inv(K_intrinsic))
print "INFO: Gimbal correction homography H1:\n", H1_gimbal

# Reference original image dimensions (used for defining corners_4 before transformation)
w_ref_original = 3816 
h_ref_original = 2138 
corners_ref = np.array([[1,1], [w_ref_original,1],[1,h_ref_original],[w_ref_original,h_ref_original]], dtype=np.float32)

# Lens distortion parameters (Theia camera calibration)
lens_mtx = np.array([[2.69065481e+04, 0.00000000e+00, 2.74787941e+03],
                     [0.00000000e+00, 2.68572137e+04, 1.57485262e+03],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
lens_dist = np.array([[-2.54857725e+00, -1.90431520e+02, -1.70987623e-03,
                       -1.14733905e-02,  2.66256271e+04]])
print "INFO: Using Lens Camera Matrix (mtx):\n", lens_mtx
print "INFO: Using Lens Distortion Coefficients (dist):\n", lens_dist


# --- Calculate transformation and output canvas size based on gimbal homography H1_gimbal ---
transformed_corners_H1 = cv2.perspectiveTransform(corners_ref.reshape((-1,1,2)), H1_gimbal)

if transformed_corners_H1 is None:
    print "ERROR: cv2.perspectiveTransform with H1_gimbal returned None. Check H1_gimbal calculation."
    sys.exit(1)

finite_transformed_corners_H1 = transformed_corners_H1[np.isfinite(transformed_corners_H1).all(axis=(1,2))]
if finite_transformed_corners_H1.shape[0] < 4:
    print "ERROR: Not enough finite corner points after H1_gimbal transformation. Check H1_gimbal and R_calib values."
    min_x_H1, max_x_H1, min_y_H1, max_y_H1 = 0, w_ref_original, 0, h_ref_original # Fallback
else:
    min_x_H1 = np.min(finite_transformed_corners_H1[:,0,0])
    max_x_H1 = np.max(finite_transformed_corners_H1[:,0,0])
    min_y_H1 = np.min(finite_transformed_corners_H1[:,0,1])
    max_y_H1 = np.max(finite_transformed_corners_H1[:,0,1])

offset_x_H1_canvas = 0.0
offset_y_H1_canvas = 0.0
if min_x_H1 < 0: offset_x_H1_canvas = -min_x_H1
if min_y_H1 < 0: offset_y_H1_canvas = -min_y_H1

# Canvas dimensions for the gimbal warp stage
# Note: cv2.warpPerspective dsize is (width, height)
gimbal_warp_canvas_width = int(np.ceil(max_x_H1 - min_x_H1 + (offset_x_H1_canvas if min_x_H1 >=0 else 0) )) # if min_x_H1 is already positive, no need to add offset_x_H1_canvas to width based on it.
gimbal_warp_canvas_height = int(np.ceil(max_y_H1 - min_y_H1 + (offset_y_H1_canvas if min_y_H1 >=0 else 0) )) # Similar logic for height

# Simpler canvas calculation: width = max_coord + offset_if_min_was_negative
gimbal_warp_canvas_width = int(np.ceil(max_x_H1 + offset_x_H1_canvas))
gimbal_warp_canvas_height = int(np.ceil(max_y_H1 + offset_y_H1_canvas))


if gimbal_warp_canvas_width <=0: gimbal_warp_canvas_width = w_ref_original
if gimbal_warp_canvas_height <=0: gimbal_warp_canvas_height = h_ref_original

offset_matrix_for_H1_warp = np.array([[1,0, offset_x_H1_canvas],[0,1, offset_y_H1_canvas],[0,0,1]], dtype=np.float32)
H_final_gimbal_warp = np.dot(offset_matrix_for_H1_warp, H1_gimbal)
print "INFO: Gimbal warp canvas dimensions (W x H): {} x {}".format(gimbal_warp_canvas_width, gimbal_warp_canvas_height)

# --- Determine Final Cropping Parameters ---
# Transform original reference corners by the complete H_final_gimbal_warp to find their positions on the warp canvas
corners_on_gimbal_warp_canvas = cv2.perspectiveTransform(corners_ref.reshape((-1,1,2)), H_final_gimbal_warp)
if corners_on_gimbal_warp_canvas is None:
    print "ERROR: perspectiveTransform for final crop corners failed. Using full canvas for crop."
    w1_crop, w2_crop, h1_crop, h2_crop = 0, gimbal_warp_canvas_width, 0, gimbal_warp_canvas_height
else:
    finite_crop_corners = corners_on_gimbal_warp_canvas[np.isfinite(corners_on_gimbal_warp_canvas).all(axis=(1,2))]
    if finite_crop_corners.shape[0] >= 4:
        sorted_x_crop = np.sort(finite_crop_corners[:,0,0])
        sorted_y_crop = np.sort(finite_crop_corners[:,0,1])
        w1_crop = int(np.ceil(sorted_x_crop[1]))
        w2_crop = int(np.floor(sorted_x_crop[2]))
        h1_crop = int(np.ceil(sorted_y_crop[1]))
        h2_crop = int(np.floor(sorted_y_crop[2]))

        w1_crop = max(0, w1_crop)
        h1_crop = max(0, h1_crop)
        w2_crop = min(gimbal_warp_canvas_width, w2_crop)
        h2_crop = min(gimbal_warp_canvas_height, h2_crop)

        if w1_crop >= w2_crop or h1_crop >= h2_crop:
            print "WARNING: Invalid crop dimensions calculated (w1={}, w2={}, h1={}, h2={}). Defaulting to full gimbal warped image.".format(w1_crop, w2_crop, h1_crop, h2_crop)
            w1_crop, w2_crop, h1_crop, h2_crop = 0, gimbal_warp_canvas_width, 0, gimbal_warp_canvas_height
    else:
        print "WARNING: Not enough finite points for cropping. Using full gimbal warped image."
        w1_crop, w2_crop, h1_crop, h2_crop = 0, gimbal_warp_canvas_width, 0, gimbal_warp_canvas_height

print "INFO: Final crop region on gimbal warp canvas: x({}-{}), y({}-{})".format(w1_crop, w2_crop, h1_crop, h2_crop)

# --- Main processing loop ---
base_image_path_len = len(os.path.join(args.image_path, '')) # Length of the input directory path

for fname in images:
    print "\nINFO: Processing [{} / {}]: {}".format(counter + 1, len(images), fname)
    
    img = cv2.imread(fname)
    if img is None:
        print "WARNING: Could not read image {}. Skipping.".format(fname)
        continue

    h_current_img, w_current_img = img.shape[:2]
    print "INFO: Current image '{}' dimensions (H x W): {} x {}".format(os.path.basename(fname), h_current_img, w_current_img)
    
    # --- Lens Undistortion ---
    print "INFO: Applying lens undistortion..."
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(lens_mtx, lens_dist, (w_current_img, h_current_img), 1, (w_current_img, h_current_img))
    dst_undistorted = cv2.undistort(img, lens_mtx, lens_dist, None, newcameramtx)
    
    x_roi, y_roi, w_roi, h_roi = roi
    print "INFO: Undistortion ROI (x,y,w,h): {},{},{},{}".format(x_roi, y_roi, w_roi, h_roi)
    dst_cropped_after_undistort = dst_undistorted[y_roi : y_roi + h_roi, x_roi : x_roi + w_roi]
    
    if dst_cropped_after_undistort.size == 0:
        print "WARNING: Image became empty after undistortion and ROI crop for {}. Skipping.".format(fname)
        continue
    print "INFO: Dimensions after undistort & ROI crop: {} x {}".format(dst_cropped_after_undistort.shape[1], dst_cropped_after_undistort.shape[0])
        
    # --- Gimbal Correction Warp ---
    print "INFO: Applying gimbal correction warp..."
    dst_gimbal_warped = cv2.warpPerspective(dst_cropped_after_undistort, H_final_gimbal_warp, (gimbal_warp_canvas_width, gimbal_warp_canvas_height))
    print "INFO: Dimensions after gimbal warp: {} x {}".format(dst_gimbal_warped.shape[1], dst_gimbal_warped.shape[0])
 
    # --- Final Crop ---
    print "INFO: Applying final crop..."
    dst_final_crop = dst_gimbal_warped[h1_crop:h2_crop , w1_crop:w2_crop]
    
    if dst_final_crop.size == 0:
        print "WARNING: Image became empty after final crop for {}. Skipping.".format(fname)
        continue
    print "INFO: Dimensions after final crop: {} x {}".format(dst_final_crop.shape[1], dst_final_crop.shape[0])

    relative_fname = os.path.basename(fname) # Simpler way to get just the filename
    output_file_path = os.path.join(calibrated_output_dir, relative_fname)
    
    output_file_dir = os.path.dirname(output_file_path) # Should just be calibrated_output_dir
    if not os.path.exists(output_file_dir): # Should already exist, but good practice
        os.makedirs(output_file_dir)

    try:
        cv2.imwrite(output_file_path, dst_final_crop)
        print "INFO: Saved calibrated image to: {}".format(output_file_path)
    except cv2.error as e:
        print "ERROR: Failed to write image {}: {}".format(output_file_path, e)
    except Exception as e:
        print "ERROR: An unexpected error occurred while writing file {}: {}".format(output_file_path, e)

    counter += 1
 
cv2.destroyAllWindows()
print "\n--- Calibration process complete ---"
print "Successfully processed and saved {} out of {} images.".format(counter, len(images))
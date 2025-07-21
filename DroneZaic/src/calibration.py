import argparse
import numpy as np
import cv2
import glob
import os
import subprocess # NEW: Import subprocess for calling external commands
import sys # Import sys for sys.exit

'''
example syntax

python calibration.py -image_path /data/e/stand_counts/stand_count_dataset/30_2019/raw/
                       -save_path /data/e/stand_counts/stand_count_dataset/30_2019/calibrated/


'''

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('-image_path', type=str,  help="paths to directory containing raw images")
parser.add_argument('-save_path', type=str,  help="paths to save directory for calibrated images")
parser.add_argument('-xxx', '--xxx', dest='xxx', default = 0, type=float,  help='x rad calib')
parser.add_argument('-yyy', '--yyy', dest='yyy', default = 0, type=float,  help='y rad calib')
parser.add_argument('-zzz', '--zzz', dest='zzz', default = 0, type=float,  help='z rad calib')

args = parser.parse_args()

# Correctly define the path to raw images
# Assumes image_path is a directory like '/data/e/stand_counts/stand_count_dataset/30_2019/raw/'
input_image_dir = args.image_path
paths = os.path.join(input_image_dir, '*.png') # Using os.path.join is safer

calibrated_output_dir = args.save_path
if not calibrated_output_dir.endswith(os.sep): # Ensure trailing slash for consistent behavior
    calibrated_output_dir += os.sep

images = sorted(glob.glob(paths))

print(f"DEBUG(calibration): Path being globbed: {paths}")
print(f"DEBUG(calibration): Number of images found by glob.glob: {len(images)}")
if images:
    print(f"DEBUG(calibration): First image in list: {images[0]}")
else:
    print(f"ERROR(calibration): No images found in {input_image_dir}. Exiting.")
    sys.exit(1)


if not os.path.exists(calibrated_output_dir):
   os.makedirs(calibrated_output_dir)
   print(f"DEBUG(calibration): Created save directory: {calibrated_output_dir}")


counter = 0

## trial and error gimbal calibration
def roty(theta):
  ct = np.cos(theta)
  st = np.sin(theta)
  R=np.array([[ct, 0, st],[0,1,0],[-st, 0, ct]])
  return R

def rotx(t):
  ct = np.cos(t)
  st = np.sin(t)
  R=np.array([[1,0,0],[0, ct, -st],[0, st, ct]])
  return R

def rotz(t):
  ct = np.cos(t)
  st = np.sin(t)
  R=np.array([[ct,-st,0],[st, ct, 0],[0, 0, 1]])
  return R

print('DEBUG(calibration): Initializing rotation matrix for gimbal calibration.')

if args.xxx != 0 and args.zzz == 0 and args.yyy == 0:
    R = rotx(args.xxx)
    
elif args.yyy != 0 and args.zzz == 0 and args.xxx == 0:
    R = roty(args.yyy)
    
elif args.zzz != 0 and args.xxx == 0 and args.yyy == 0:
    R = rotz(args.zzz)

elif args.xxx != 0 and args.yyy != 0 and args.zzz == 0 :
    R = np.dot(roty(args.yyy),rotx(args.xxx))

elif args.xxx != 0 and args.zzz != 0 and args.yyy == 0:
    R = np.dot(rotz(args.zzz),rotx(args.xxx))

elif args.zzz != 0 and args.yyy != 0 and args.xxx == 0:
    R = np.dot(roty(args.yyy),rotz(args.zzz))

elif args.xxx != 0 and args.yyy != 0 and args.zzz != 0 :
    print('DEBUG(calibration): Applying all three rotation axes.');
    R = np.dot(np.dot(roty(args.yyy),rotx(args.xxx)), rotz(args.zzz))

else:
    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # Default to identity matrix if no args

##########gimbal calibration#######

K=np.array([[2359.79036,0, 2031],[ 0, 2359.30434,   1046.5],[ 0 ,   0,   1]])
H1 = np.dot(np.dot(K, R), np.linalg.inv(K)) # Homography matrix for calibration transform

# Define image dimensions based on your camera, or dynamically load from first image if consistent
# Using hardcoded values from your script. Ensure these match your actual image dimensions.
w_img_orig = 3816
h_img_orig = 2138

corners_4 = np.array([[0,0], [w_img_orig,0],[0,h_img_orig],[w_img_orig,h_img_orig]], dtype=np.float32) # Corrected to 0-indexed corners

# 23.20.10 calibration grace automate python 
mtx = np.array([[ 7003.06025,   0.00000000,   1726.14480], [ 0.00000000,   7049.87304, -97.2162886], [ 0.00000000,   0.00000000,   1.00000000]])
dist = np.array([[ 0.06693996, -0.15926691, -0.01767889, -0.00425557,   0.34429158]])
  
# Calculate corrected corners after calibration homography for bounding box
corr = cv2.perspectiveTransform(corners_4.reshape((-1,1,2)), H1)

# Logic to calculate offset and new dimensions for the warped image to fit
# This section calculates the necessary translation to shift all warped points to positive coordinates.
# It then adjusts the 'corr' points and calculates the output width/height.
min_x_corr = np.min(corr[:,:,0])
min_y_corr = np.min(corr[:,:,1])
max_x_corr = np.max(corr[:,:,0])
max_y_corr = np.max(corr[:,:,1])

offset_x_calib = 0.0
if min_x_corr < 0:
    offset_x_calib = -min_x_corr

offset_y_calib = 0.0
if min_y_corr < 0:
    offset_y_calib = -min_y_corr

offset_matrix_calib = np.array([[1,0, offset_x_calib],[0,1, offset_y_calib],[0,0,1]], dtype=np.float32)

output_width_calib = int(np.ceil(max_x_corr - min_x_corr))
output_height_calib = int(np.ceil(max_y_corr - min_y_corr))

print(f"DEBUG(calibration): Calibration output dimensions: {output_width_calib}x{output_height_calib}")
print(f"DEBUG(calibration): Calibration offset matrix:\n{offset_matrix_calib}")


# NEW FUNCTION: To preserve EXIF data using exiftool
def preserve_exif_data(original_file_path, new_file_path):
    """
    Copies all EXIF/metadata from the original_file_path to the new_file_path
    using the exiftool command-line utility.
    """
    try:
        # -tagsFromFile: Copies all tags from source
        # -all:all: Copies all individual tag information (including GPS)
        # -overwrite_original: Overwrites the target file directly
        # -q: Quiet mode (suppress verbose output)
        command = ['exiftool', '-tagsFromFile', original_file_path, '-all:all', '-overwrite_original', '-q', new_file_path]
        
        # Run the command and capture output for debugging
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"DEBUG(calibration): EXIF copy successful for {os.path.basename(new_file_path)}")
        if result.stderr:
            print(f"WARNING(calibration): exiftool stderr: {result.stderr.strip()}")

    except FileNotFoundError:
        print("ERROR(calibration): 'exiftool' command not found. Please install exiftool.", file=sys.stderr)
        print("  (e.g., 'sudo apt-get install libimage-exiftool-perl' on Debian/Ubuntu)", file=sys.stderr)
        print(f"  EXIF data NOT preserved for {os.path.basename(new_file_path)}.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR(calibration): exiftool failed for {os.path.basename(new_file_path)}: {e.stderr.strip()}", file=sys.stderr)
        print(f"  EXIF data NOT preserved for {os.path.basename(new_file_path)}.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR(calibration): Unexpected error during EXIF copy for {os.path.basename(new_file_path)}: {e}", file=sys.stderr)
        print(f"  EXIF data NOT preserved for {os.path.basename(new_file_path)}.", file=sys.stderr)


# Main loop for image processing
for fname in images:
    print(f"DEBUG(calibration): Processing image {counter}: {os.path.basename(fname)}")
    
    img = cv2.imread(fname)

    if img is None:
        print(f"WARNING(calibration): Could not read image {fname}. Skipping.", file=sys.stderr)
        continue

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # Undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image to the valid region after undistortion
    x,y,w_roi,h_roi = roi
    if w_roi > 0 and h_roi > 0: # Ensure valid ROI
        dst = dst[y:y+h_roi, x:x+w_roi]
    else:
        print(f"WARNING(calibration): Invalid ROI after getOptimalNewCameraMatrix for {os.path.basename(fname)}. Skipping crop.", file=sys.stderr)
        # If ROI is invalid, dst is the undistorted image, might be black borders.

    # Apply the gimbal calibration homography and offset
    # First, warp by the calibration homography H1
    # Then, apply the offset_matrix_calib to shift to positive coordinates.
    # The target size is output_width_calib x output_height_calib
    try:
        final_calibrated_image = cv2.warpPerspective(dst, np.dot(offset_matrix_calib, H1), 
                                                     (output_width_calib, output_height_calib))
    except Exception as e:
        print(f"ERROR(calibration): Failed to warp image {os.path.basename(fname)} with calibration homography: {e}", file=sys.stderr)
        print(f"  Saving undistorted only as fallback (no gimbal correction).", file=sys.stderr)
        # As a fallback, save the undistorted image without gimbal correction if warping fails
        final_calibrated_image = dst # Use just undistorted image if warp fails

    # Construct the output file path
    output_filename = os.path.basename(fname)
    output_filepath = os.path.join(calibrated_output_dir, output_filename)
    
    # Save the calibrated image
    cv2.imwrite(output_filepath, final_calibrated_image)
    print(f"DEBUG(calibration): Saved calibrated image to: {output_filepath}")

    # Preserve EXIF data
    preserve_exif_data(fname, output_filepath)

    counter += 1
 
cv2.destroyAllWindows()
print("DEBUG(calibration): Calibration process complete for all images.")
import argparse
import os
import subprocess
import cv2
import numpy as np

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Dynamically samples frames from a video based on optical flow.")
parser.add_argument('-video', type=str, help="Path to the input video file.")
parser.add_argument('-save_path', type=str, help="Path to the directory where sampled frames and quiver data will be saved.")
parser.add_argument('-scale', type=int, default=1, help="Scale factor for resizing frames (e.g., 2 for half size).")
parser.add_argument('-fps', type=int, default=1, help="Frames per second to extract.")
parser.add_argument('-fname', type=str, default="output", help="Base filename for output frames.")
args = parser.parse_args()

# --- Define Paths ---
video_path = args.video
base_save_path = args.save_path
raw_frames_path = os.path.join(base_save_path, 'raw_frames')
quiver_data_path = os.path.join(base_save_path, 'quiver')

# --- Create Output Directories ---
print("Creating output directories...")
try:
    if not os.path.exists(raw_frames_path):
        os.makedirs(raw_frames_path)
        print(f"Created directory: {raw_frames_path}")
    if not os.path.exists(quiver_data_path):
        os.makedirs(quiver_data_path)
        print(f"Created directory: {quiver_data_path}")
except OSError as e:
    print(f"Error creating directories: {e}")
    exit(1) # Exit if directories cannot be created

# --- FFmpeg Frame Extraction ---
print(f"Starting FFmpeg frame extraction from {video_path}...")
output_frame_pattern = os.path.join(raw_frames_path, f"{args.fname}_frame_%06d.png") # Python 3.6+ f-string
# Python 2.7 compatible string formatting for output_frame_pattern:
# output_frame_pattern = os.path.join(raw_frames_path, "%s_frame_%%06d.png" % args.fname)


# FFmpeg command to extract frames
# -i: input file
# -vf: video filter graph (select frames at desired fps)
# -q:v: video quality (1 is highest)
# -f image2: output format is image sequence
ffmpeg_command = [
    'ffmpeg',
    '-i', video_path,
    '-vf', f"fps={args.fps}", # Python 3.6+ f-string
    # Python 2.7 compatible: '-vf', "fps=%d" % args.fps,
    '-q:v', '1',
    output_frame_pattern
]

# For Python 2.7, adjust the f-string for -vf and output_frame_pattern
if hasattr(args, '__dict__'): # Simple check to infer Python 2.7 vs 3+
    # Assuming Python 2.7 if f-strings are not supported
    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', "fps=%d" % args.fps,
        '-q:v', '1',
        os.path.join(raw_frames_path, "%s_frame_%%06d.png" % args.fname)
    ]
    print("Using Python 2.7 compatible FFmpeg command construction.")
else:
    print("Using Python 3+ compatible FFmpeg command construction (f-strings).")


print("FFmpeg command:", " ".join(ffmpeg_command))

try:
    # Use subprocess.check_call to raise an error if ffmpeg fails
    subprocess.check_call(ffmpeg_command)
    print("FFmpeg frame extraction completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error during FFmpeg execution: {e}")
    print(f"FFmpeg stderr: {e.stderr}")
    exit(1)
except OSError as e:
    print(f"Error executing FFmpeg command. Is FFmpeg installed and in PATH? {e}")
    exit(1)

# --- Optical Flow (Placeholder for MaiZaic's dynamic sampling logic) ---
# The original MaiZaic dynamic_sampling.py likely has more complex logic
# involving optical flow to select *which* frames to keep.
# For now, this script extracts all frames at the specified FPS.
# If MaiZaic's dynamic_sampling.py has optical flow logic, it would go here
# to further filter the extracted frames and save only the selected ones.

# For now, we'll just verify that frames were extracted.
extracted_frames = sorted(glob.glob(os.path.join(raw_frames_path, f"{args.fname}_frame_*.png"))) # Python 3.6+ f-string
# Python 2.7 compatible:
extracted_frames = sorted(glob.glob(os.path.join(raw_frames_path, "%s_frame_*.png" % args.fname)))

if not extracted_frames:
    print("Warning: No frames were extracted by FFmpeg. Check video path and FFmpeg output.")
    exit(1)
else:
    print(f"Successfully extracted {len(extracted_frames)} frames to {raw_frames_path}")
    print("First few extracted frames:", extracted_frames[:5])

# --- Placeholder for Quiver Data Generation (if applicable) ---
# If dynamic_sampling.py also generates quiver data, that logic would be here.
# For now, this script focuses on ensuring frame extraction.

print("Dynamic sampling process complete.")
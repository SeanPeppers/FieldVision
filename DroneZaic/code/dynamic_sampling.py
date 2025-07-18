import cv2
import argparse
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import time
import re
import piexif
from PIL import Image
import subprocess

def parse_srt_file(srt_file):
    timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})")
    gps_pattern = re.compile(r"\[latitude\s*:\s*([-+]?[0-9]*\.?[0-9]+)\].*\[longtitude\s*:\s*([-+]?[0-9]*\.?[0-9]+)\]")
    
    timestamps, gps_data = [], []
    
    try:
        with open(srt_file, "r") as file:
            lines = file.readlines()
        
        current_timestamp = None
        for line in lines:
            timestamp_match = timestamp_pattern.search(line)
            gps_match = gps_pattern.search(line)
            
            if timestamp_match:
                current_timestamp = (timestamp_match.group(1), timestamp_match.group(2))
            
            if gps_match and current_timestamp:
                latitude, longitude = float(gps_match.group(1)), float(gps_match.group(2))
                timestamps.append(current_timestamp)
                gps_data.append((latitude, longitude))
    except FileNotFoundError:
        print(f"SRT file not found: {srt_file}. Continuing without GPS data.")
        return [], []
    except Exception as e:
        print(f"Error parsing SRT file {srt_file}: {e}. Continuing without GPS data.")
        return [], []
    
    return timestamps, gps_data

def find_closest_gps(frame_time_in_ms, timestamps, gps_data):
    frame_timedelta = timedelta(milliseconds=frame_time_in_ms)
    
    for i, (start_time_str, end_time_str) in enumerate(timestamps):
        start_h, start_m, start_s_ms = start_time_str.replace(',', '.').split(':')
        start_s, start_ms = start_s_ms.split('.')
        start_timedelta = timedelta(hours=int(start_h), minutes=int(start_m), seconds=int(start_s), milliseconds=int(start_ms))

        end_h, end_m, end_s_ms = end_time_str.replace(',', '.').split(':')
        end_s, end_ms = end_s_ms.split('.')
        end_timedelta = timedelta(hours=int(end_h), minutes=int(end_m), seconds=int(end_s), milliseconds=int(end_ms))
        
        if start_timedelta <= frame_timedelta <= end_timedelta:
            return gps_data[i]
    
    return (None, None)

def embed_gps_metadata(image_path, latitude, longitude):
    if latitude is None or longitude is None:
        print(f"Skipping GPS metadata embedding for {image_path} (No GPS data available)")
        return

    lat_ref = "N" if latitude >= 0 else "S"
    lon_ref = "E" if longitude >= 0 else "W"

    command = [
        "exiftool",
        f"-GPSLatitude={abs(latitude)}",
        f"-GPSLatitudeRef={lat_ref}",
        f"-GPSLongitude={abs(longitude)}",
        f"-GPSLongitudeRef={lon_ref}",
        "-overwrite_original",
        image_path
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    result_code = process.returncode

    if result_code == 0:
        print(f"Embedded GPS via exiftool into {image_path}: Lat {latitude}, Lon {longitude}")
    else:
        print(f"Failed to embed GPS via exiftool into {image_path}. Error: {stderr.decode().strip()}")
        
def save_frame_with_gps(frame, frame_path, frame_time_ms, timestamps, gps_data):
    latitude, longitude = find_closest_gps(frame_time_ms, timestamps, gps_data)
    
    cv2.imwrite(frame_path, frame)
    embed_gps_metadata(frame_path, latitude, longitude)
    print(f"Saved frame {frame_path} with embedded GPS (Lat {latitude}, Lon {longitude})")
    
    return frame_path
    
def detect_cam_movement_video(video_path, srt_file, save_path, scale, start_number, fps, win_size, ss, img_format, live_mode, live_delay):
    translation_threshold = 5
    quiver_path = os.path.join(save_path, 'quiver')
    distribution_path = os.path.join(save_path, 'distribution')
    raw_path = os.path.join(save_path, 'raw')

    if not os.path.exists(quiver_path):
        os.makedirs(quiver_path)
    if not os.path.exists(distribution_path):
        os.makedirs(distribution_path)
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)

    if srt_file:
        timestamps, gps_data = parse_srt_file(srt_file)
    else:
        timestamps, gps_data = [], []

    print("Parsed SRT timestamps count:", len(timestamps))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {video_fps}, Total Frames: {total_frames}")

    if video_fps == 0:
        print("Warning: Video FPS is 0, cannot calculate default interval. Setting to 30 frames.")
        default_interval_frames = 30
    else:
        default_interval_frames = int(video_fps / fps) if fps > 0 else int(video_fps)
    if default_interval_frames == 0:
        default_interval_frames = 1 
    
    degree_mean = []

    # --- Initial frame processing for both modes ---
    # The first frame is handled outside the main loop to initialize prev_gray.
    
    # If not in live mode, set the video position for initial frame.
    if not live_mode and ss > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, ss * 1000)
    
    ret, init_frame = cap.read()
    if not ret:
        print("Failed to read first frame. Exiting.")
        return

    initial_frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    
    # Use start_number for the first frame saved.
    flname_initial = os.path.join(raw_path, f"{args.fname}_frame_{start_number:06d}.{img_format}")

    if timestamps and gps_data:
        save_frame_with_gps(init_frame, flname_initial, initial_frame_time_ms, timestamps, gps_data)
    else:
        cv2.imwrite(flname_initial, init_frame)
        print(f"Saved frame {flname_initial} without GPS metadata")
    
    prev_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape

    new_height = int(np.round(height/scale))
    new_width = int(np.round(width/scale))

    prev_gray = cv2.resize(prev_gray, (new_width, new_height))

    # `i` is the counter for frames that are *actually saved*
    i = start_number + 1 
    
    # Variables for non-live mode (original logic)
    current_interval = default_interval_frames
    j_original = 0 # Corresponds to the 'j' in the original non-live logic
    non_translation_index_original = 0 # Corresponds to 'non_translation_index' in original non-live logic

    # Variables for live mode
    frames_since_last_non_translation_save_live = 0 # For controlling saves in live non-translation
    j_skip_overlap_live = 0 # For controlling skips in live high-overlap

    print("Starting frame processing loop...")
    while True:
        if live_mode:
            # Live mode: Introduce delay and read sequentially
            if live_delay > 0:
                time.sleep(live_delay)
            ret, frame = cap.read()
            if not ret:
                print('End of video stream.')
                break
            current_frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_video_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        else: # Not live mode: Original seeking logic
            # Calculate the frame index to seek to
            frame_to_seek = (current_video_frame_number if 'current_video_frame_number' in locals() else cap.get(cv2.CAP_PROP_POS_FRAMES)) + current_interval
            
            # Ensure not to seek beyond total frames if known
            if total_frames > 0 and frame_to_seek >= total_frames:
                print('Reached end of video file during non-live processing.')
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_seek)
            ret, frame = cap.read()
            if not ret: # Can happen if seeking past end or read error
                print('End of video stream or seek failed during non-live processing.')
                break
            current_frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_video_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray_original, (new_width, new_height))

        t = time.time() 
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 150, win_size, 3, 5, 1.2, 0)
        elapsed_flow_time = time.time()-t

        flow_x = flow[:,:,0]
        flow_y = flow[:,:,1]

        overlap = compute_overlap(np.abs(flow_x), np.abs(flow_y))
        translation = evaluate_trajectory(np.abs(flow_x), np.abs(flow_y), translation_threshold)

        save_this_frame = False

        if live_mode:
            # --- Live Mode Logic ---
            if not translation:
                frames_since_last_non_translation_save_live += 1
                j_skip_overlap_live = 0

                if frames_since_last_non_translation_save_live == 1:
                    save_this_frame = True
                    print(f'Live Mode: Non-translation detected. Saving frame {i} (first non-translating frame).')
                elif default_interval_frames > 0 and (current_video_frame_number % int(video_fps / fps / 2)) == 0:
                    save_this_frame = True
                    print(f'Live Mode: Still non-translating. Saving frame {i} at reduced frequency.')
                else:
                    print(f'Live Mode: Non-translating. Skipping frame {current_video_frame_number} as per interval.')
            elif translation:
                frames_since_last_non_translation_save_live = 0
                
                if 0.85 <= overlap <= 0.98:
                    save_this_frame = True
                    j_skip_overlap_live = 0
                    print(f'Live Mode: Translating with good overlap ({overlap*100:.2f}%), saving frame {i}.')
                elif overlap > 0.98:
                    j_skip_overlap_live += 1
                    if j_skip_overlap_live >= 4:
                        save_this_frame = True
                        j_skip_overlap_live = 0
                        print(f'Live Mode: High overlap ({overlap*100:.2f}%), forced save of frame {i} after skips.')
                    else:
                        print(f'Live Mode: High overlap ({overlap*100:.2f}%), skipping frame {current_video_frame_number} (skip count: {j_skip_overlap_live}).')
                else:
                    save_this_frame = True
                    j_skip_overlap_live = 0
                    print(f'Live Mode: Low overlap ({overlap*100:.2f}%), saving frame {i} due to significant change.')
        
        else: # --- Non-Live Mode Logic (original logic) ---
            if not translation or non_translation_index_original == 1:
                if non_translation_index_original == 0:
                    current_interval = np.floor(current_interval/2)
                    # Note: In non-live mode, we 'go back' by adjusting the current_video_frame_number
                    # for the next iteration's cap.set.
                    current_video_frame_number -= current_interval # Adjust the logical position for next iteration
                    non_translation_index_original += 1
                    print(f'Non-Live Mode: Not translating, moving logical index back to {current_video_frame_number}, current interval is {current_interval}.')
                    continue # Skip saving and re-evaluate from the 'moved back' frame.
                
                # If non_translation_index_original is already 1, or after the first rollback
                current_interval = np.floor(current_interval) # Ensure it's integer for frame skipping
                save_this_frame = True
                print(f'Non-Live Mode: Still not translating, saving frame {i}. Current interval is {current_interval}.')
                j_original = 0 # Reset skip counter
                non_translation_index_original += 1

            elif translation:
                current_interval = default_interval_frames
                non_translation_index_original = 0
                
                if 0.85 <= overlap <= 0.98:
                    save_this_frame = True
                    print(f'Non-Live Mode: Translating with good overlap ({overlap*100:.2f}%), saving frame {i}.')
                    j_original = 0
                elif overlap > 0.98:
                    j_original += 1
                    if j_original >= 4:
                        save_this_frame = True
                        j_original = 0
                        print(f'Non-Live Mode: High overlap ({overlap*100:.2f}%), forced save of frame {i} after skips.')
                    else:
                        print(f'Non-Live Mode: High overlap ({overlap*100:.2f}%), skipping frame {current_video_frame_number} (skip count: {j_original}).')
                else: # overlap < 0.85
                    current_interval = np.ceil(current_interval/2) # Increase sampling rate by halving interval
                    # If this makes interval 1, we save. Otherwise, we re-evaluate closer.
                    if current_interval <= 1: # If interval becomes 1, save the current frame.
                         save_this_frame = True
                         print(f'Non-Live Mode: Low overlap ({overlap*100:.2f}%), forcing save of frame {i} due to high change (interval = {current_interval}).')
                    else:
                        # Move back in original mode to resample more densely
                        current_video_frame_number -= current_interval 
                        print(f'Non-Live Mode: Low overlap ({overlap*100:.2f}%), moving logical index back to {current_video_frame_number} to re-evaluate denser (new interval = {current_interval}).')
                        continue # Re-evaluate from the 'moved back' frame.
                    j_original = 0 # Reset skip counter


        if save_this_frame:
            flname_to_save = os.path.join(raw_path, f"{args.fname}_frame_{i:06d}.{img_format}")
            if timestamps and gps_data:
                save_frame_with_gps(frame, flname_to_save, current_frame_time_ms, timestamps, gps_data)
            else:
                cv2.imwrite(flname_to_save, frame)
                print(f"Saved frame {flname_to_save} without GPS metadata")
            
            prev_gray = gray.copy()
            mean_direction_in_degree = compute_direction_save_plots(flow_x, flow_y, flow, i, quiver_path, distribution_path, args.fname) 
            degree_mean.append(mean_direction_in_degree)
            i += 1
            # For non-live mode, if a frame was saved, the next frame to *check* will be based on the interval
            # This is implicit in how cap.set(CAP_PROP_POS_FRAMES) works with `current_interval`
            # For live mode, `cap.read()` naturally progresses.

    cap.release()

    degree_diff = compute_degree_diff(np.array(degree_mean))
    print("******")
    print("Degree differences between saved frames:", degree_diff)
    degree_plot(np.array(degree_diff), os.path.join(save_path, "plot_of_diff_angle.png"))
    
    with open(os.path.join(save_path, f'{args.fname}_angle_diff.csv'), 'a') as f1:
        wr = csv.writer(f1, quoting = csv.QUOTE_NONE)
        for angle_difff in degree_diff:
            wr.writerow([angle_difff])

def compute_direction_save_plots(flow_x, flow_y, flow, index, quiver_path, distribution_path, fname):
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    mean_magnitude = compute_iqr_average(magnitude)

    direction = np.arctan2(flow_y, flow_x)
    direction_degree = ((np.degrees(direction)+360)%360)
    mean_directions = compute_iqr_average(direction)
    mean_direction_in_degree = ((np.degrees(mean_directions)+360)%360)

    draw_quiver(flow, magnitude, 50, os.path.join(quiver_path, f"{fname}_quiver_frame_{index:06d}.png"))
    pixel_distribution_plot(magnitude, direction_degree, os.path.join(distribution_path, f"{fname}_distribution_frame_{index:06d}.png"))

    return mean_direction_in_degree
      
def draw_quiver(flow, magnitude, skip, save_path):
    filename_match = re.search(r'_frame_(\d{6})', save_path)
    if filename_match:
        filename_int = int(filename_match.group(1))
        prev_filename_int = filename_int - 1
    else:
        filename_int = 0
        prev_filename_int = -1

    height, width = flow.shape[:2]
    y, x = np.mgrid[0:height:skip, 0:width:skip]
    
    min_color = 0
    max_color = 100

    tick_positions = np.linspace(min_color, max_color, int(max_color/10)+1)

    plt.figure(figsize=(10, 8))
    plt.quiver(x, y, flow[y, x, 0], flow[y, x, 1], magnitude[y, x], pivot='mid', cmap=plt.cm.jet, linewidth=5, headwidth=3)
    plt.colorbar(label='Magnitude', ticks=tick_positions, format='%.2f', orientation='vertical', shrink=0.57)
    plt.clim(min_color, max_color)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Quiver from Frame {prev_filename_int:03d} to {filename_int:03d} (plot every {skip}th vector)')

    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    
    plt.savefig(save_path)
    plt.close()

def compute_iqr_average(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    data_iqr = np.mean(data[np.logical_and(data>=q1,data<=q3)])
    return data_iqr

def evaluate_trajectory(x, y, thresh):
    magnitude = np.sqrt(x**2 + y**2)
    height, width = magnitude.shape

    inner = magnitude[2*int(height/6):height-2*int(height/6), 2*int(width/6):width-2*int(width/6)]
    avg_inner = np.mean(inner, axis=(0,1))

    top_rec = np.mean(magnitude[:int(height/6),:], axis=(0,1))
    bottom_rec = np.mean(magnitude[height-int(height/6):, :], axis=(0,1))
    left_rec = np.mean(magnitude[int(height/6):height-int(height/6), :int(width/6)], axis=(0,1))
    right_rec = np.mean(magnitude[int(height/6):height-int(height/6), width-int(width/6):], axis=(0,1))
    
    avg_outer = (top_rec+bottom_rec+left_rec+right_rec)/4

    if np.abs(avg_inner-avg_outer)<thresh:
        translation = True
    else:
        translation = False
    return translation

def compute_overlap(x, y):
    height, width = x.shape
    displacement_x = compute_iqr_average(x)
    displacement_y = compute_iqr_average(y)

    overlap_width = width-displacement_x
    overlap_height = height-displacement_y
    
    overlap_percentage = (overlap_width*overlap_height)/(width*height)
    return overlap_percentage 

def compute_degree_diff(mean_degree1):
    mean_degree = np.asarray(mean_degree1)
    degree_diff_array = []
    
    for i in range(1, len(mean_degree)):
        diff = abs(mean_degree[i] - mean_degree[i-1])
        if diff > 180:
            min_direction_diff = 360 - diff
        else:
            min_direction_diff = diff
        
        degree_diff_array.append(min_direction_diff)
    return degree_diff_array

def pixel_distribution_plot(magnitude, direction, save_path):
    mag = magnitude.flatten()
    dirctn = direction.flatten()

    plt.figure(figsize=(10,8))
    
    plt.hist(mag, bins=100, density = False, color='blue', alpha=0.5, label='Magnitude')
    plt.hist(dirctn, bins = 100, density = False, color='red', alpha =0.5, label='Direction')

    plt.xlim(0,360)
    plt.ylim(0, mag.shape[0])
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Distribution of Magnitude and Direction')

    plt.savefig(save_path)
    plt.close()

def compute_mad(data):
    median = np.median(data)
    absolute_deviation = np.abs(data-median)
    mad = np.median(absolute_deviation)
    return mad

def degree_plot(degrees, save_path):
    x = np.arange(len(degrees))

    plt.figure(figsize=(10, 10))
    plt.plot(x, degrees)

    plt.xlabel('frame no')
    plt.ylabel('degree')
    plt.title('Plot of the degree angle between frames')

    plt.savefig(save_path)
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('-image_path', type=str, help="paths to one or more images or image directories")
    parser.add_argument('-srt', type=str, default= None, help="path to corresponding .srt metadata file.")
    parser.add_argument('-video', type=str, help="paths to one video")
    parser.add_argument('-save_path', type=str, dest='save_path', default="RESULTS/global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), help="path to save result")
    parser.add_argument('-hm', type=str, help='txt file that stores homography matrices')
    parser.add_argument('-scale', type=int, dest='scale', default=1, help='the downsampled scale for the frame')
    parser.add_argument('-fps', type=float, dest='fps', default=0.5, help='desired frames per second for sampling (after movement detection)')
    parser.add_argument('-win', type=int, dest='win_size', default=50, help='window size for optical flow')
    parser.add_argument('-start_number', type=int, dest='start_number', default=1, help='initial number to save the frame id')
    parser.add_argument('-ss', type=int, default=0, help='start time in seconds to begin video processing')
    parser.add_argument('-fname', type=str, help="desired prefix name for frame extracted")
    parser.add_argument('-format', type=str, choices=['jpg', 'tif', 'png'], default='tif', help="image format to save frames (jpg, tif, or png).")
    
    # New arguments for live mode
    parser.add_argument('--live_mode', action='store_true', help="Enable live video stream simulation mode.")
    parser.add_argument('--live_delay', type=float, default=0.01, help="Simulated delay (in seconds) between processing frames in live mode.")

    args = parser.parse_args()
    
    start_time_overall = datetime.now()
    detect_cam_movement_video(args.video, args.srt, args.save_path, args.scale, args.start_number, args.fps, args.win_size, args.ss, args.format, args.live_mode, args.live_delay)
    elapsed_overall = (datetime.now() - start_time_overall).total_seconds()

    print(f'Dynamic sampling process completed in: {elapsed_overall:.2f} seconds.')
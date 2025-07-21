"""
author: Dewi Kharismawati
project: MaiZaic
call this by:
- for video
python dynamic_sampling.py -video /path/to/raw/video -save_path /path/to/where/you/want/to/save/the/quiver/plot
example:
python dynamic_sampling/dynamic_sampling.py -video /media/dek8v5/f/aerial_imaging/images/23r/grace/23.6/DJI_0205.MOV -save_path /home/dek8v5/Documents/cornetv2/data_ori/FINAL_CORNETV2_DATASET/1_gps_jpg_23r_06_23_205_seedling_parallel_1pass/jpeg -srt /media/dek8v5/f/aerial_imaging/images/23r/grace/23.6/DJI_0205.SRT -win 100 -scale 3 -fname 23r_06_23 -format jpg -real_time_streaming
- for frames
python dynamic_sampling.py -image_path /path/to/raw/images -save_path /path/to/where/you/want/to/save/the/quiver/plot
"""

import cv2
import argparse
from datetime import datetime
import numpy as np
import os
import matplotlib
import shutil # Import shutil for directory removal

matplotlib.use("agg")
import matplotlib.pyplot as plt
import csv
import time
import re
import piexif
from PIL import Image
import subprocess


def parse_srt_file(srt_file):
    timestamp_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})"
    )
    gps_pattern = re.compile(
        r"\[latitude\s*:\s*([-+]?[0-9]*\.?[0-9]+)\].*\[longtitude\s*:\s*([-+]?[0-9]*\.?[0-9]+)\]"
    )

    timestamps, gps_data = [], []

    with open(srt_file, "r") as file:
        lines = file.readlines()

    current_timestamp = None
    for line in lines:
        timestamp_match = timestamp_pattern.search(line)
        gps_match = gps_pattern.search(line)

        if timestamp_match:
            current_timestamp = (
                timestamp_match.group(1),
                timestamp_match.group(2),
            )

        if gps_match and current_timestamp:
            latitude, longitude = float(gps_match.group(1)), float(gps_match.group(2))
            timestamps.append(current_timestamp)
            gps_data.append((latitude, longitude))

    print("Extracted timestamps:", timestamps)
    print("Extracted GPS data:", gps_data)
    return timestamps, gps_data


def find_closest_gps(frame_time, timestamps, gps_data):
    frame_seconds = sum(
        float(x) * 60**i
        for i, x in enumerate(reversed(frame_time.replace(",", ".").split(":")))
    )

    for i, (start_time, end_time) in enumerate(timestamps):
        start_seconds = sum(
            float(x) * 60**i
            for i, x in enumerate(reversed(start_time.replace(",", ".").split(":")))
        )
        end_seconds = sum(
            float(x) * 60**i
            for i, x in enumerate(reversed(end_time.replace(",", ".").split(":")))
        )

        if start_seconds <= frame_seconds <= end_seconds:
            return gps_data[i]

    return (None, None)


def convert_to_exif_format(value):
    degrees = int(abs(value))
    minutes = int((abs(value) - degrees) * 60)
    seconds = round(((abs(value) - degrees) * 60 - minutes) * 60 * 10000)
    return ((degrees, 1), (minutes, 1), (seconds, 10000))


def embed_gps_metadata(image_path, latitude, longitude):
    if latitude is None or longitude is None:
        print(
            "Skipping GPS metadata embedding for {} (No GPS data available)".format(
                image_path
            )
        )
        return

    lat_ref = "N" if latitude >= 0 else "S"
    lon_ref = "E" if longitude >= 0 else "W"

    command = [
        "exiftool",
        "-GPSLatitude={}".format(abs(latitude)),
        "-GPSLatitudeRef={}".format(lat_ref),
        "-GPSLongitude={}".format(abs(longitude)),
        "-GPSLongitudeRef={}".format(lon_ref),
        "-overwrite_original",
        image_path,
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    result_code = process.returncode

    if result_code == 0:
        print(
            "Embedded GPS via exiftool into {}: Lat {}, Lon {}".format(
                image_path, latitude, longitude
            )
        )
    else:
        print(
            "Failed to embed GPS via exiftool into {}. Error: {}".format(
                image_path, stderr
            )
        )


def save_frame_with_gps(frame, frame_path, frame_time_sec, timestamps, gps_data):
    frame_time_formatted = datetime.utcfromtimestamp(frame_time_sec).strftime(
        "%H:%M:%S,%f"
    )[:-3]
    latitude, longitude = find_closest_gps(frame_time_formatted, timestamps, gps_data)

    cv2.imwrite(frame_path, frame)
    embed_gps_metadata(frame_path, latitude, longitude)
    print(
        "Saved frame {} with embedded GPS (Lat {}, Lon {})".format(
            frame_path, latitude, longitude
        )
    )

    return frame_path

def clear_output_directories(save_path):
    """
    Removes the quiver, distribution, and raw directories within save_path if they exist.
    """
    quiver_path = os.path.join(save_path, "quiver")
    distribution_path = os.path.join(save_path, "distribution")
    raw_path = os.path.join(save_path, "raw")

    for path in [quiver_path, distribution_path, raw_path]:
        if os.path.exists(path):
            print(f"Clearing directory: {path}")
            shutil.rmtree(path)
        else:
            print(f"Directory not found, skipping clearing: {path}")


def detect_cam_movement_video(
    video, srt_file, save_path, scale, i, fps, win_size, ss, img_format, real_time_streaming, clear_output
):
    if clear_output:
        clear_output_directories(save_path)

    translation_threshold = 5
    quiver_path = os.path.join(save_path, "quiver")
    distribution_path = os.path.join(save_path, "distribution")
    raw_path = os.path.join(save_path, "raw")

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

    print(timestamps)
    print(gps_data)

    video_capture = cv2.VideoCapture(video)

    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    if real_time_streaming and original_fps > 0:
        frame_time_in_seconds = 1.0 / original_fps
        print(f"Simulating real-time streaming with a delay of {frame_time_in_seconds:.4f} seconds per frame.")
    else:
        frame_time_in_seconds = 0


    current_fps = fps

    default_interval = int(video_capture.get(cv2.CAP_PROP_FPS) * current_fps)
    current_interval = default_interval
    if ss == 0:
        frame_index = default_interval
    else:
        frame_index = ss * 30

    ret = video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, init_frame = video_capture.read()

    frame_time_sec = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    flname = os.path.join(raw_path, args.fname + "_frame_%06d.%s" % (i, img_format))

    if ret:
        if timestamps and gps_data:
            save_frame_with_gps(
                init_frame, flname, frame_time_sec, timestamps, gps_data
            )
            i += 1
        else:
            cv2.imwrite(flname, init_frame)
            print("Saved frame {} without GPS metadata".format(flname))
            i += 1
    else:
        print("The video is empty!")
        return

    prev_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape

    new_height = int(np.round(height / scale))
    new_width = int(np.round(width / scale))

    prev_gray = cv2.resize(prev_gray, (new_width, new_height))

    j = 0
    non_translation_index = 0
    degree_mean = []

    translation = True

    while True:
        start_processing_time = time.time()

        print("================================")

        print("frame index", frame_index)

        print("skip index counter j : ", j)

        ret = video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = video_capture.read()

        frame_time_sec = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if not ret:
            print("End of video")
            break
        flname = os.path.join(raw_path, args.fname + "_frame_%06d.%s" % (i, img_format))

        print("Frame time in seconds right now: ", frame_time_sec)

        gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray_original, (new_width, new_height))

        t = time.time()
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 150, win_size, 3, 5, 1.2, 0
        )
        elapsed_flow_time = time.time() - t

        print("********Elapsed time for optical flow: ", elapsed_flow_time)

        print("fps: ", fps)
        print("win_size: ", win_size)
        y, x = np.mgrid[0:new_height, 0:new_width]

        flow_x = flow[y, x, 0]
        flow_y = flow[y, x, 1]

        overlap = compute_overlap(np.abs(flow_x), np.abs(flow_y))

        translation = evaluate_trajectory(
            np.abs(flow_x), np.abs(flow_y), translation_threshold
        )

        print("translation is ", translation)

        if translation == False or non_translation_index == 1:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

            if non_translation_index == 0:
                current_interval = np.floor(current_interval / 2)
                frame_index = frame_index - current_interval + 1
                non_translation_index += 1

                print(
                    "It is not translation, moving the index back at %d current interval is %d"
                    % (frame_index, current_interval)
                )

                continue

            frame_index = frame_index + current_interval

            print("It is not translation, current interval is %d" % current_interval)
            j = 0

            if timestamps and gps_data:
                save_frame_with_gps(frame, flname, frame_time_sec, timestamps, gps_data)
                i += 1
            else:
                cv2.imwrite(flname, frame)
                print("Saved frame {} without GPS metadata".format(flname))
                i += 1

            prev_gray = gray.copy()

            mean_direction_in_degree = compute_direction_save_plots(
                flow_x, flow_y, flow, i, quiver_path, distribution_path, args.fname
            )

            degree_mean.append(mean_direction_in_degree)

            non_translation_index += 1

            end_processing_time = time.time()
            processing_duration = end_processing_time - start_processing_time
            if real_time_streaming and processing_duration < frame_time_in_seconds:
                time.sleep(frame_time_in_seconds - processing_duration)

            continue

        elif translation == True:
            print("Default interval: ", default_interval)
            current_interval = default_interval
            non_translation_index = 0

        if 0.85 <= overlap <= 0.98:

            print("This frame has %f overlap" % (overlap * 100))

            if timestamps and gps_data:
                save_frame_with_gps(frame, flname, frame_time_sec, timestamps, gps_data)
                i += 1
            else:
                cv2.imwrite(flname, frame)
                print("Saved frame {} without GPS metadata".format(flname))
                i += 1

            frame_index += default_interval

            prev_gray = gray.copy()

            mean_direction_in_degree = compute_direction_save_plots(
                flow_x, flow_y, flow, i, quiver_path, distribution_path, args.fname
            )

            degree_mean.append(mean_direction_in_degree)

            j = 0

        elif 0.98 < overlap <= 1:
            print("Overlap is over:  %f" % (overlap * 100))
            print("Reducing the sampling rate")

            frame_index = frame_index + np.floor(default_interval / 2)

            if j >= 4:
                j = 0
                i += 1

                if timestamps and gps_data:
                    save_frame_with_gps(
                        frame, flname, frame_time_sec, timestamps, gps_data
                    )
                    i += 1
                else:
                    cv2.imwrite(flname, frame)
                    print("Saved frame {} without GPS metadata".format(flname))
                    i += 1

                prev_gray = gray.copy()

                mean_direction_in_degree = compute_direction_save_plots(
                    flow_x, flow_y, flow, i, quiver_path, distribution_path, args.fname
                )

                degree_mean.append(mean_direction_in_degree)

                end_processing_time = time.time()
                processing_duration = end_processing_time - start_processing_time
                if real_time_streaming and processing_duration < frame_time_in_seconds:
                    time.sleep(frame_time_in_seconds - processing_duration)

                continue

            j += 1

        else:
            print("Overlap is lower:  %f" % (overlap * 100))
            print("Current time increasing the sampling rate")
            current_interval = np.ceil(current_interval / 2)
            if (current_interval) > 1:
                frame_index = frame_index - current_interval
            else:
                print("Uh oh, current frame is the adjacent of the previous frame")

                if timestamps and gps_data:
                    save_frame_with_gps(
                        frame, flname, frame_time_sec, timestamps, gps_data
                    )
                    i += 1
                else:
                    cv2.imwrite(flname, frame)
                    print("Saved frame {} without GPS metadata".format(flname))
                    i += 1

                prev_gray = gray.copy()

                mean_direction_in_degree = compute_direction_save_plots(
                    flow_x, flow_y, flow, i, quiver_path, distribution_path, args.fname
                )

                degree_mean.append(mean_direction_in_degree)
                j = 0
        end_processing_time = time.time()
        processing_duration = end_processing_time - start_processing_time
        if real_time_streaming and processing_duration < frame_time_in_seconds:
            time.sleep(frame_time_in_seconds - processing_duration)


    degree_diff = compute_degree_diff(np.array(degree_mean))
    print("******")
    print(degree_diff)
    degree_plot(
        np.array(degree_diff), os.path.join(save_path, "plot_of_diff_angle.png")
    )

    with open(args.save_path + "/" + args.fname + "_angle_diff.csv", "a") as f1:
        for angle_difff in degree_diff:
            wr = csv.writer(f1, quoting=csv.QUOTE_NONE)
            wr.writerow([angle_difff])


def compute_direction_save_plots(
    flow_x, flow_y, flow, index, quiver_path, distribution_path, fname
):

    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    mean_magnitude = compute_iqr_average(magnitude)

    direction = np.arctan2(flow_y, flow_x)
    direction_degree = (np.degrees(direction) + 360) % 360
    mean_directions = compute_iqr_average(direction)
    mean_direction_in_degree = (np.degrees(mean_directions) + 360) % 360

    draw_quiver(
        flow,
        magnitude,
        50,
        os.path.join(quiver_path, fname + "_quiver_frame_%06d.png" % index),
    )

    pixel_distribution_plot(
        magnitude,
        direction_degree,
        os.path.join(distribution_path, fname + "_distribution_frame_%06d.png" % index),
    )

    return mean_direction_in_degree


def draw_quiver(flow, magnitude, skip, save_path):

    filename = int(save_path[-10:-4])
    prev_filename = int(filename) - 1

    height, width = flow.shape[:2]
    y, x = np.mgrid[0:height:skip, 0:width:skip]

    min_color = 0
    max_color = 100

    tick_positions = np.linspace(min_color, max_color, int(max_color / 10) + 1)

    plt.figure(figsize=(10, 8))
    plt.quiver(
        x,
        y,
        flow[y, x, 0],
        flow[y, x, 1],
        magnitude[y, x],
        pivot="mid",
        cmap=plt.cm.jet,
        linewidth=5,
        headwidth=3,
    )
    plt.colorbar(
        label="Magnitude",
        ticks=tick_positions,
        format="%.2f",
        orientation="vertical",
        shrink=0.57,
    )
    plt.clim(min_color, max_color)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(
        "Quiver from Frame %.3d to %.3d (plot every %dth vector)"
        % (prev_filename, filename, skip)
    )

    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")

    plt.savefig(save_path)
    plt.close()


def compute_iqr_average(data):

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    data_iqr = np.mean(data[np.logical_and(data >= q1, data <= q3)])

    return data_iqr


def evaluate_trajectory(x, y, thresh):

    magnitude = np.sqrt(x**2 + y**2)
    height, width = magnitude.shape

    inner = magnitude[
        2 * int(height / 6) : height - 2 * int(height / 6),
        2 * int(width / 6) : width - 2 * int(width / 6),
    ]

    avg_inner = np.mean(inner, axis=(0, 1))

    top_rec = np.mean(magnitude[: int(height / 6), :], axis=(0, 1))
    bottom_rec = np.mean(magnitude[height - int(height / 6) :, :], axis=(0, 1))
    left_rec = np.mean(
        magnitude[int(height / 6) : height - int(height / 6), : int(width / 6)],
        axis=(0, 1),
    )
    right_rec = np.mean(
        magnitude[int(height / 6) : height - int(height / 6), width - int(width / 6) :],
        axis=(0, 1),
    )

    avg_outer = (top_rec + bottom_rec + left_rec + right_rec) / 4

    print("diff ", np.abs(avg_inner - avg_outer))
    if np.abs(avg_inner - avg_outer) < thresh:
        translation = True
    else:
        translation = False

    return translation


def compute_overlap(x, y):
    height, width = x.shape
    displacement_x = compute_iqr_average(x)
    displacement_y = compute_iqr_average(y)

    overlap_width = width - displacement_x
    overlap_height = height - displacement_y

    overlap_percentage = (overlap_width * overlap_height) / (width * height)

    return overlap_percentage


def compute_degree_diff(mean_degree1):

    mean_degree = np.asarray(mean_degree1)
    degree_diff_array = []

    for i in range(1, len(mean_degree)):

        diff = abs(mean_degree[i] - mean_degree[i - 1])
        if diff > 180:
            min_direction_diff = 360 - diff
        else:
            min_direction_diff = diff

        degree_diff_array.append(min_direction_diff)

    return degree_diff_array


def pixel_distribution_plot(magnitude, direction, save_path):

    mag = magnitude.flatten()
    dirctn = direction.flatten()

    plt.figure(figsize=(10, 8))

    plt.hist(mag, bins=100, density=False, color="blue", alpha=0.5, label="Magnitude")

    plt.hist(dirctn, bins=100, density=False, color="red", alpha=0.5, label="Direction")

    plt.xlim(0, 360)
    plt.ylim(0, mag.shape[0])
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Distribution of Magnitude and Direction")

    plt.savefig(save_path)
    plt.close()


def compute_mad(data):

    median = np.median(data)

    absolute_deviation = np.abs(data - median)

    mad = np.median(absolute_deviation)
    return mad


def degree_plot(degrees, save_path):
    x = np.arange(len(degrees))
    plt.figure(figsize=(10, 10))
    plt.plot(x, degrees)
    plt.xlabel("frame no")
    plt.ylabel("degree")
    plt.title("plot of the degree angle between frames")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-image_path", type=str, help="paths to one or more images or image directories"
    )
    parser.add_argument(
        "-srt", type=str, default=None, help="path to corresponding .srt metadata file."
    )
    parser.add_argument("-video", type=str, help="paths to one video")
    parser.add_argument(
        "-save_path",
        type=str,
        dest="save_path",
        default="RESULTS/global_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="path to save result",
    )
    parser.add_argument(
        "-hm", type=str, help="txt file that stores homography matrices"
    )
    parser.add_argument(
        "-scale",
        type=int,
        dest="scale",
        default=1,
        help="the downsampled scale for the frame",
    )
    parser.add_argument(
        "-fps",
        type=float,
        dest="fps",
        default=0.5,
        help="the downsampled scale for the frame",
    )
    parser.add_argument(
        "-win",
        type=int,
        dest="win_size",
        default=50,
        help="the downsampled scale for the frame",
    )
    parser.add_argument(
        "-start_number",
        type=int,
        dest="start_number",
        default=1,
        help="initial number to save the frame id",
    )
    parser.add_argument(
        "-ss", type=int, default=0, help="where do you want the start time to extract"
    )
    parser.add_argument(
        "-fname", type=str, help="desired prefix name for frame extracted"
    )
    parser.add_argument(
        "-format",
        type=str,
        choices=["jpg", "tif", "png"],
        default="tif",
        help="image format to save frames (jpg, tif, or png).",
    )
    parser.add_argument(
        "-real_time_streaming",
        action="store_true",
        help="Simulate real-time video streaming by introducing a delay between frames.",
    )
    parser.add_argument(
        "-clear_output",
        action="store_true",
        help="Clear existing output directories before running.",
    )
    args = parser.parse_args()

    start_time = datetime.now()
    detect_cam_movement_video(
        args.video,
        args.srt,
        args.save_path,
        args.scale,
        args.start_number,
        args.fps,
        args.win_size,
        args.ss,
        args.format,
        args.real_time_streaming,
        args.clear_output,
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    print("dynamic sampling time elapsed: ", elapsed)
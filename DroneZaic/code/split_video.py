import subprocess
import re
from datetime import datetime, timedelta
import os

# Define base paths relative to the /app directory
BASE_APP_DIR = "/app"
INPUT_OUTPUT_DIR = os.path.join(BASE_APP_DIR, "input_videos") 

def get_video_duration(video_path):
    """Gets the duration of a video file using ffprobe as a timedelta object."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_seconds = float(result.stdout.strip())
        
        return timedelta(milliseconds=int(duration_seconds * 1000))
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting video duration for {video_path}: {e}")
        print("Ensure FFmpeg/ffprobe is installed and accessible in your system's PATH.")
        return None

def parse_time(time_str):
    """Parses an SRT time string (HH:MM:SS,ms) into a timedelta object."""
    time_str = time_str.replace('.', ',') 
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))

def format_time_srt(td_object):
    """Formats a timedelta object back into an SRT time string (HH:MM:SS,ms)."""
    total_seconds = int(td_object.total_seconds())
    milliseconds = int(td_object.microseconds / 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def format_time_ffmpeg(td_object):
    """Formats a timedelta object into an FFmpeg-compatible time string (HH:MM:SS.ms)."""
    total_seconds = int(td_object.total_seconds())
    milliseconds = int(td_object.microseconds / 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def split_srt_with_metadata(input_srt_path, output_srt1_path, output_srt2_path, split_timedelta):
    """Splits an SRT file with embedded metadata into two, adjusting timestamps for the second file."""
    subtitles = []
    current_subtitle_block = []

    try:
        with open(input_srt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: # Empty line indicates end of a subtitle block
                    if current_subtitle_block:
                        subtitles.append(current_subtitle_block)
                        current_subtitle_block = []
                else:
                    current_subtitle_block.append(line)
            if current_subtitle_block: 
                subtitles.append(current_subtitle_block)
    except FileNotFoundError:
        print(f"Error: SRT file not found at '{input_srt_path}'")
        return False
    except Exception as e:
        print(f"Error reading SRT file '{input_srt_path}': {e}")
        return False

    part1_subs = []
    part2_subs = []
    
    for block_lines in subtitles:
        if not block_lines: 
            continue

        try:
            sub_number = int(block_lines[0])
        except ValueError:
            print(f"Warning: Could not parse subtitle number from line: '{block_lines[0]}'. Skipping block.")
            continue

        if len(block_lines) < 2:
            print(f"Warning: Subtitle block {sub_number} is too short, missing timestamps. Skipping.")
            continue
        
        time_line_match = re.match(r'(\d{2}:\d{2}:\d{2}[.,]\d{3}) --> (\d{2}:\d{2}:\d{2}[.,]\d{3})', block_lines[1])
        if not time_line_match:
            print(f"Warning: Could not parse time line for subtitle {sub_number}: '{block_lines[1]}'. Skipping block.")
            continue
            
        start_time_str, end_time_str = time_line_match.groups()
        start_time = parse_time(start_time_str) 
        end_time = parse_time(end_time_str)

        if start_time < split_timedelta:
            part1_subs.append(block_lines)
        else:
            adjusted_block = [block_lines[0]] 
            
            adjusted_start_time = start_time - split_timedelta
            adjusted_end_time = end_time - split_timedelta
            
            adjusted_start_time = max(timedelta(0), adjusted_start_time)
            adjusted_end_time = max(timedelta(0), adjusted_end_time)

            adjusted_block.append(f"{format_time_srt(adjusted_start_time)} --> {format_time_srt(adjusted_end_time)}")
            adjusted_block.extend(block_lines[2:]) 
            part2_subs.append(adjusted_block)
            
    try:
        with open(output_srt1_path, 'w', encoding='utf-8') as f1:
            for i, block_lines in enumerate(part1_subs):
                f1.write(f"{i + 1}\n")
                for line_idx, line in enumerate(block_lines):
                    if line_idx == 0:
                        continue 
                    f1.write(f"{line}\n")
                f1.write("\n")

        with open(output_srt2_path, 'w', encoding='utf-8') as f2:
            for i, block_lines in enumerate(part2_subs):
                f2.write(f"{i + 1}\n")
                for line_idx, line in enumerate(block_lines):
                    if line_idx == 0:
                        continue
                    f2.write(f"{line}\n")
                f2.write("\n")
        return True
    except Exception as e:
        print(f"Error writing SRT files: {e}")
        return False

def split_video(video_path, output_video1_path, output_video2_path, split_time_str_ffmpeg):
    """Splits a video file using FFmpeg."""
    try:
        cmd1 = [
            'ffmpeg', '-i', video_path, '-t', split_time_str_ffmpeg, '-c', 'copy', output_video1_path
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)

        cmd2 = [
            'ffmpeg', '-i', video_path, '-ss', split_time_str_ffmpeg, '-c', 'copy', output_video2_path
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error splitting video with FFmpeg for {video_path}:")
        print(f"Command: {' '.join(e.cmd)}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        print("Ensure FFmpeg is installed and accessible in your system's PATH and the video file is valid.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during video splitting for {video_path}: {e}")
        return False

def main():
    mov_filename = "DJI_0604.MOV"
    srt_filename = "dji.srt"

    # Input and output paths are now the same directory
    input_mov_path = os.path.join(INPUT_OUTPUT_DIR, mov_filename)
    input_srt_path = os.path.join(INPUT_OUTPUT_DIR, srt_filename)

    # Ensure the input directory exists (it should, as the files are there)
    os.makedirs(INPUT_OUTPUT_DIR, exist_ok=True) 

    mov_name_base = os.path.splitext(mov_filename)[0]
    srt_name_base = os.path.splitext(srt_filename)[0]

    # Output files will be placed in the same directory as inputs
    output_mov_part1 = os.path.join(INPUT_OUTPUT_DIR, f"{mov_name_base}_part1.MOV")
    output_mov_part2 = os.path.join(INPUT_OUTPUT_DIR, f"{mov_name_base}_part2.MOV")
    output_srt_part1 = os.path.join(INPUT_OUTPUT_DIR, f"{srt_name_base}_part1.srt")
    output_srt_part2 = os.path.join(INPUT_OUTPUT_DIR, f"{srt_name_base}_part2.srt")

    print(f"Processing: {input_mov_path} and {input_srt_path}")

    video_duration = get_video_duration(input_mov_path)
    if video_duration is None:
        print("Could not determine video duration. Exiting.")
        return

    midpoint_timedelta = video_duration / 2
    
    midpoint_time_str_ffmpeg = format_time_ffmpeg(midpoint_timedelta)
    midpoint_time_str_srt = format_time_srt(midpoint_timedelta)

    print(f"Video duration: {format_time_srt(video_duration)}, calculated midpoint: {midpoint_time_str_srt}")

    print(f"Splitting video '{input_mov_path}'...")
    if split_video(input_mov_path, output_mov_part1, output_mov_part2, midpoint_time_str_ffmpeg):
        print(f"Video split successfully into '{output_mov_part1}' and '{output_mov_part2}'.")
    else:
        print("Video splitting failed. Exiting.")
        return 

    print(f"Splitting SRT '{input_srt_path}'...")
    if split_srt_with_metadata(input_srt_path, output_srt_part1, output_srt_part2, midpoint_timedelta):
        print(f"SRT split successfully into '{output_srt_part1}' and '{output_srt_part2}'.")
    else:
        print("SRT splitting failed. Exiting.")

if __name__ == "__main__":
    main()
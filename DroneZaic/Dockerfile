# Use an official Python runtime as a parent image
FROM python:3.5-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Define the command to run when the container starts
# This entire CMD is now one very long, escaped string.
# Each logical command within the script is terminated by a semicolon.
CMD ["bash", "-c", "\
INPUT_VIDEOS_DIR=\"/app/input_videos\"; \
EXTRACTED_FRAMES_DIR=\"/app/extracted_frames\"; \
OUTPUT_DIR=\"/app/outputs\"; \
echo \"--- Docker Container Startup ---\"; \
echo \"Current working directory: $(pwd)\"; \
echo \"Input videos will be read from: $INPUT_VIDEOS_DIR\"; \
echo \"Extracted frames will be stored in: $EXTRACTED_FRAMES_DIR\"; \
echo \"Stitched outputs will be saved to: $OUTPUT_DIR\"; \
if [ ! -d \"$INPUT_VIDEOS_DIR\" ]; then echo \"ERROR: Input videos directory '$INPUT_VIDEOS_DIR' does not exist inside the container.\"; echo \"       Please ensure your Docker volume mount for input_videos is correct.\"; exit 1; fi; \
if [ -z \"$(ls -A \"$INPUT_VIDEOS_DIR\")\" ]; then echo \"WARNING: Input videos directory '$INPUT_VIDEOS_DIR' is empty. No videos to process.\"; echo \"         Please ensure you have placed video files in your mounted input_videos folder.\"; exit 0; fi; \
echo \"Proceeding with DroneZaic pipeline...\"; \
mkdir -p \"$EXTRACTED_FRAMES_DIR\" || { echo \"ERROR: Failed to create $EXTRACTED_FRAMES_DIR\"; exit 1; }; \
mkdir -p \"$OUTPUT_DIR\" || { echo \"ERROR: Failed to create $OUTPUT_DIR\"; exit 1; }; \
VIDEO_COUNT=0; \
for video_file in \"$INPUT_VIDEOS_DIR\"/*; do \
    if [ -f \"$video_file\" ]; then \
        VIDEO_COUNT=$((VIDEO_COUNT+1)); \
        echo \"--- Processing video: $(basename \"$video_file\") ---\"; \
        VIDEO_FRAMES_SUBDIR=\"${EXTRACTED_FRAMES_DIR}/$(basename \"${video_file%.*}\")_frames\"; \
        mkdir -p \"$VIDEO_FRAMES_SUBDIR\" || { echo \"ERROR: Failed to create $VIDEO_FRAMES_SUBDIR\"; exit 1; }; \
        echo \"Extracting frames from $(basename \"$video_file\")...\"; \
        ffmpeg -i \"$video_file\" -vf fps=1 -q:v 2 \"${VIDEO_FRAMES_SUBDIR}/%05d.jpg\" 2>&1 || { echo \"ERROR: FFmpeg failed for $(basename \"$video_file\")\"; exit 1; }; \
        echo \"Frames extracted to: $VIDEO_FRAMES_SUBDIR\"; \
        echo \"Running DroneZaic stitcher on frames from $(basename \"$video_file\")...\"; \
        CURRENT_OUTPUT_IMAGE=\"${OUTPUT_DIR}/stitched_$(basename \"${video_file%.*}\").jpg\"; \
        python code/stitcher.py -image_path \"$VIDEO_FRAMES_SUBDIR\" -save_path \"$CURRENT_OUTPUT_IMAGE\" 2>&1 || { echo \"ERROR: DroneZaic stitcher failed for frames from $(basename \"$video_file\")\"; exit 1; }; \
        echo \"Stitching complete. Output: $CURRENT_OUTPUT_IMAGE\"; \
    else \
        echo \"WARNING: Skipping non-file item in $INPUT_VIDEOS_DIR: $(basename \"$video_file\")\"; \
    fi; \
done; \
if [ \"$VIDEO_COUNT\" -eq 0 ]; then echo \"No valid video files found for processing in $INPUT_VIDEOS_DIR.\"; fi; \
echo \"--- DroneZaic pipeline finished. ---\"\
"]
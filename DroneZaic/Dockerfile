# Use a more recent and actively supported Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    ffmpeg \
    exiftool && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt first to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all *necessary* code files.
# With the .dockerignore, this will only bring in the actual Python scripts and __init__.py files.
COPY . .

# --- START SUPER-VERBOSER DIAGNOSTIC AND VALIDATION STEPS ---
# These steps are designed to FAIL THE BUILD if files/imports/definitions are not as expected.
# This should pinpoint the exact issue if file copying or content is wrong.

# Step 1: Verify core file presence and __init__.py structure
RUN echo "--- VERIFYING CORE FILE PRESENCE AND STRUCTURE ---" && \
    ls -l /app/code/__init__.py || (echo "ERROR: /app/code/__init__.py missing!" && exit 1) && \
    ls -l /app/code/asift/__init__.py || (echo "ERROR: /app/code/asift/__init__.py missing!" && exit 1) && \
    ls -l /app/code/asift/common.py || (echo "ERROR: /app/code/asift/common.py missing!" && exit 1) && \
    ls -l /app/code/asift/find_obj.py || (echo "ERROR: /app/code/asift/find_obj.py missing!" && exit 1) && \
    ls -l /app/code/asift/asift.py || (echo "ERROR: /app/code/asift/asift.py missing!" && exit 1) && \
    ls -l /app/code/mini_mosaic_360.py || (echo "ERROR: /app/code/mini_mosaic_360.py missing!" && exit 1)

# Step 2: Print content of crucial Python files for visual inspection in logs
RUN echo "\n--- CONTENT OF /app/code/asift/common.py ---" && cat /app/code/asift/common.py
RUN echo "\n--- CONTENT OF /app/code/asift/find_obj.py ---" && cat /app/code/asift/find_obj.py
RUN echo "\n--- CONTENT OF /app/code/asift/asift.py ---" && cat /app/code/asift/asift.py
RUN echo "\n--- CONTENT OF /app/code/mini_mosaic_360.py ---" && cat /app/code/mini_mosaic_360.py

# Step 3: Verify import lines in asift.py
RUN echo "\n--- VERIFYING IMPORTS IN /app/code/asift/asift.py ---" && \
    grep -E "from \.common import Timer" /app/code/asift/asift.py || \
        (echo "ERROR: 'from .common import Timer' not found in asift.py!" && exit 1) && \
    grep -E "from \.find_obj import init_feature, filter_matches, explore_match" /app/code/asift/asift.py || \
        (echo "ERROR: 'from .find_obj import ...' not found in asift.py!" && exit 1)

# Step 4: Verify import line in mini_mosaic_360.py
RUN echo "\n--- VERIFYING IMPORT IN /app/code/mini_mosaic_360.py ---" && \
    grep -E "from code.asift.asift import my_asift" /app/code/mini_mosaic_360.py || \
        (echo "ERROR: 'from code.asift.asift import my_asift' not found in mini_mosaic_360.py!" && exit 1)

# Step 5: Verify init_feature definition in find_obj.py
RUN echo "\n--- VERIFYING 'init_feature' DEFINITION IN /app/code/asift/find_obj.py ---" && \
    grep -E "^def init_feature\(" /app/code/asift/find_obj.py || \
        (echo "ERROR: 'init_feature' function definition not found in find_obj.py! Check for typos or if it's nested." && exit 1)

# --- END SUPER-VERBOSER DIAGNOSTIC AND VALIDATION STEPS ---


# IMPORTANT: Explicitly add /app to PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Add execute permission to the maizaic_run.sh script
RUN chmod +x code/maizaic_run.sh

# Define the command to run when the container starts.
CMD ["code/maizaic_run.sh", "-h", "surf", "-d", "false"]
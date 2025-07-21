# Use a more recent and actively supported Python runtime as a parent image
# Changed from buster to bookworm (Debian 12, current stable)
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies:
# Added libgdal-dev and gdal-bin for rasterio/GDAL support
# ADDED cmake for OpenCV contrib compilation support
# Ensure git is installed for cloning repositories
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    ffmpeg \
    exiftool \
    libgdal-dev \
    gdal-bin \
    cmake \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt first to leverage Docker's build cache
COPY requirements.txt .

# Upgrade pip and setuptools before installing requirements
RUN pip install --no-cache-dir --upgrade pip setuptools

# Install core Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- SuperGlue and LightGlue Installation ---
# Clone the SuperGluePretrainedNetwork repository.
# Removed 'pip install -e' as it's not a standard Python package.
RUN git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git

# Clone the LightGlue repository and install it
# We'll clone into a specific directory, then navigate and install
RUN git clone https://github.com/cvg/LightGlue.git /app/LightGlue \
    && pip install --no-cache-dir -e /app/LightGlue

# Copy all *necessary* src files.
# With the .dockerignore, this will only bring in the actual Python scripts and __init__.py files.
COPY . .

# IMPORTANT: Explicitly add /app and the SuperGlue directory to PYTHONPATH
# Initialize PYTHONPATH if it might be unset
ENV PYTHONPATH=/app:/app/SuperGluePretrainedNetwork:${PYTHONPATH:-}

RUN chmod +x src/maizaic_run.sh
RUN chmod +x src/run_global.sh
RUN chmod +x src/run_mini.sh
RUN chmod +x src/full_pipeline.sh

# Define the command to run when the container starts.
CMD ["src/maizaic_run.sh", "-h", "surf", "-d", "false"]
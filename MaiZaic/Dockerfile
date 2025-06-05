# Stage 1: Temporary stage to copy requirement.txt
FROM busybox AS requirements_stage
COPY requirement.txt /requirement.txt

# Base image for the environment
FROM ubuntu:18.04

# Set environment variables
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python2.7 \
    python-pip \
    python2.7-dev \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python2.7 as the default 'python' command
RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1

# Upgrade pip, setuptools, and wheel for Python 2.7 compatibility
RUN pip install --no-cache-dir --upgrade pip==20.3.4 && \
    pip install --no-cache-dir --upgrade setuptools==44.1.1 && \
    pip install --no-cache-dir --upgrade wheel

# Set application working directory inside container
WORKDIR /app/MaiZaic/code

# Copy requirement.txt from the temporary stage
COPY --from=requirements_stage /requirement.txt ./requirement.txt

# Modify requirements.txt:
# 1. Remove tensorflow-gpu (if present)
# 2. Add a newline, then add tensorflow==1.2.0 (CPU version)
# 3. opencv-contrib-python is handled by pip uninstall + install below
RUN sed -i '/tensorflow-gpu/d' requirement.txt && \
    echo "" >> requirement.txt && \
    echo "tensorflow==1.2.0" >> requirement.txt

# Install Python dependencies: UNINSTALL existing opencv first, then install from requirements.txt
RUN pip uninstall -y opencv-python opencv-contrib-python || true && \
    pip install --no-cache-dir -r requirement.txt

# Copy application code
COPY code/ ./

# Set default command to start a bash shell
CMD ["/bin/bash"]
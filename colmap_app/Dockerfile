# colmap_app/Dockerfile
# Use COLMAP's pre-built GPU image as the base.
# This image includes a compiled COLMAP with CUDA support.
# Using the 'latest' tag for convenience.
FROM colmap/colmap:latest

# Set the working directory in the container to /app
WORKDIR /app

# Install Python 3.11 and pip. COLMAP images might not have the specific Python version
# or pip setup that our application needs by default.
# Also install git, ffmpeg, unzip, curl for general utilities.
# Ensure DEBIAN_FRONTEND is noninteractive to prevent prompts.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    python3-dev \
    git \
    ffmpeg \
    unzip \
    curl \
    tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Configure timezone non-interactively
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



# Create a Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the application's requirements file into the container at /app
COPY requirements.txt .

# Install the application's Python dependencies into the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container at /app
COPY colmap_main.py .
COPY colmap_processor.py .

# Create necessary data directories inside the container
RUN mkdir -p data/videos data/outputs data/logs

# Expose port 8000 (the port your FastAPI app will run on)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "colmap_main:app", "--host", "0.0.0.0", "--port", "8000"]

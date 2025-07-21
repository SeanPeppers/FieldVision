# Use a lightweight official Python runtime as a parent image.
# Python 3.11 is chosen for compatibility with many modern libraries.
FROM python:3.11-slim-bullseye

# Set the working directory in the container to /app
WORKDIR /app

# Install necessary system dependencies:
# - git: For cloning the VGGT repository.
# - ffmpeg: Essential for video processing (e.g., extracting frames).
# - build-essential: Provides tools like gcc, g++ for compiling C/C++ extensions, often needed by Python packages.
# - libgl1-mesa-glx, libsm6, libxext6, libglib2.0-0: Common dependencies for OpenCV and other graphical libraries.
# - unzip: For extracting any compressed datasets or files.
# - curl: For downloading files from URLs.
# rm -rf /var/lib/apt/lists/*: Cleans up apt caches to keep the image size small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    unzip \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Clone the VGGT repository
# We clone it into a subdirectory within /app. This keeps the VGGT code separate
# from your application's main files.
RUN git clone https://github.com/facebookresearch/vggt.git /app/vggt

# Copy the application's requirements file into the container at /app
COPY requirements.txt .

# Install the application's Python dependencies
# --no-cache-dir: Prevents pip from storing cache, saving space.
# --upgrade pip: Ensures pip is up-to-date.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install VGGT itself in editable mode so its modules can be imported
WORKDIR /app/vggt
RUN pip install --no-cache-dir -e .
WORKDIR /app # Return to main working directory

# Copy the entire application source code into the container at /app
# This copies main.py and models.py directly into /app.
COPY main.py .
COPY models.py .

# Create necessary data directories inside the container
# This ensures that the application has places to store input videos, outputs, and logs.
RUN mkdir -p data/videos data/outputs data/logs

# Expose port 8000 (the port your FastAPI app will run on)
EXPOSE 8000

# Command to run the application
# uvicorn: ASGI server for FastAPI.
# main:app: Points to the 'app' FastAPI instance within 'main.py' at the root of /app.
# --host 0.0.0.0: Makes the server accessible from outside the container.
# --port 8000: Specifies the port to listen on.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

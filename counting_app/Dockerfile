# Use a lightweight official Python runtime as a parent image.
# We're using a specific version for stability. Python 3.9 is a good balance.
# Changed from 'buster' to 'bullseye' as Buster repositories might be archived.
FROM python:3.9-slim-bullseye

# Set the working directory in the container to /app
WORKDIR /app

# Install necessary system dependencies for OpenCV and Git
# libgl1-mesa-glx: Provides OpenGL functionality, crucial for OpenCV.
# libsm6, libxext6: X11 libraries often required by GUI applications or libraries like OpenCV.
# libglib2.0-0: Provides libgthread-2.0.so.0, another common dependency for OpenCV.
# git: Required to clone repositories specified in requirements.txt (e.g., ultralytics/CLIP.git).
# rm -rf /var/lib/apt/lists/*: Cleans up apt caches to keep the image size small.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libsm6 libxext6 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Don't store the cache, saving space.
# --upgrade pip: Ensure pip is up-to-date.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application folder into the container at /app
# This copies everything from your local 'app' directory to '/app' in the container.
COPY app/ app/

# Create necessary data directories inside the container
# This ensures that the application has places to store input and output files.
RUN mkdir -p data/videos data/output

# Expose port 8000 (the port your FastAPI app will run on)
EXPOSE 8000

# Command to run the application
# uvicorn: ASGI server for FastAPI.
# app.main: Points to the 'app' module and 'main.py' file.
# app: The FastAPI instance named 'app' within 'main.py'.
# --host 0.0.0.0: Makes the server accessible from outside the container.
# --port 8000: Specifies the port to listen on.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

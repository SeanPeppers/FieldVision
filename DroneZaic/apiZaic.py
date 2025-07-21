import os
import subprocess
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import uuid
import shutil

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = '/tmp/uploads' # Temporary directory for uploaded files
OUTPUT_BASE_DIR = '/app/outputs' # Your existing outputs directory
ALLOWED_EXTENSIONS = {'mov', 'mp4', 'avi', 'srt'} # Allowed file types

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_BASE_DIR'] = OUTPUT_BASE_DIR
app.config['APP_DIR'] = '/app' # Define APP_DIR for the Flask app to match the shell script's assumption

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(app.config['APP_DIR'], 'input_videos'), exist_ok=True) # Ensure input_videos dir exists

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    API endpoint to receive a video and optional SRT file,
    and trigger the mini-mosaic generation pipeline.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    srt_file = request.files.get('srt') # SRT file is optional

    if video_file.filename == '':
        return jsonify({"error": "No selected video file"}), 400

    if video_file and allowed_file(video_file.filename):
        # Create a unique directory for this request to avoid conflicts
        request_id = str(uuid.uuid4())
        request_dir = os.path.join(app.config['UPLOAD_FOLDER'], request_id)
        os.makedirs(request_dir, exist_ok=True)

        video_filename = secure_filename(video_file.filename)
        video_path_temp = os.path.join(request_dir, video_filename)
        video_file.save(video_path_temp)
        print(f"Video saved temporarily to: {video_path_temp}")

        srt_filename = None
        srt_path_temp = None
        if srt_file and allowed_file(srt_file.filename):
            srt_filename = secure_filename(srt_file.filename)
            srt_path_temp = os.path.join(request_dir, srt_filename)
            srt_file.save(srt_path_temp)
            print(f"SRT saved temporarily to: {srt_path_temp}")
        else:
            print("No valid SRT file provided.")

        # Copy the uploaded video and SRT (if any) to the /app/input_videos directory
        # where the shell script expects them.
        target_video_path = os.path.join(app.config['APP_DIR'], 'input_videos', video_filename)
        shutil.copy(video_path_temp, target_video_path)
        print(f"Copied video to: {target_video_path}")

        target_srt_path = None
        if srt_filename:
            target_srt_path = os.path.join(app.config['APP_DIR'], 'input_videos', srt_filename)
            shutil.copy(srt_path_temp, target_srt_path)
            print(f"Copied SRT to: {target_srt_path}")

        try:
            # Construct the command to run your shell script with arguments
            # UPDATED: Changed path from /app/full_pipeline.sh to /app/src/full_pipeline.sh
            command = [os.path.join(app.config['APP_DIR'], 'src', 'full_pipeline.sh'), video_filename]
            if srt_filename:
                command.append(srt_filename)

            print(f"\n--- Starting full_pipeline.sh for request {request_id} ---")
            print(f"Executing command: {' '.join(command)}")

            process = subprocess.run(
                command,
                check=True, # Raise an exception for non-zero exit codes
                cwd=app.config['APP_DIR'] # Run the script from the /app directory
            )

            print(f"--- full_pipeline.sh finished for request {request_id} ---\n")

            # After successful execution, the mini-mosaics should be in:
            # /app/outputs/asift_mini_mosaics (based on your script's HM_METHOD)
            mini_mosaics_dir = os.path.join(app.config['OUTPUT_BASE_DIR'], 'asift_mini_mosaics')

            # --- How to return the mini-mosaics ---
            # Option 1: Zip the directory and send it back
            output_zip_path = os.path.join(request_dir, f"mini_mosaics_{request_id}.zip")
            shutil.make_archive(output_zip_path.rsplit('.', 1)[0], 'zip', mini_mosaics_dir)

            # Clean up the temporary upload directory for this request
            shutil.rmtree(request_dir)
            # Clean up the copied input files from /app/input_videos
            os.remove(target_video_path)
            if target_srt_path and os.path.exists(target_srt_path):
                os.remove(target_srt_path)

            return send_file(output_zip_path, as_attachment=True, download_name=f"mini_mosaics_{request_id}.zip")

        except subprocess.CalledProcessError as e:
            print(f"\n--- Error executing script for request {request_id} ---")
            print(f"Script exited with error code: {e.returncode}")
            print(f"--- End error for request {request_id} ---\n")
            # Clean up in case of error too
            if os.path.exists(request_dir):
                shutil.rmtree(request_dir)
            if os.path.exists(target_video_path):
                os.remove(target_video_path)
            if target_srt_path and os.path.exists(target_srt_path):
                os.remove(target_srt_path)
            return jsonify({"error": "Pipeline execution failed", "details": f"Script exited with code {e.returncode}. Check server logs for details."}), 500
        except Exception as e:
            print(f"\n--- An unexpected error occurred for request {request_id} ---")
            print(f"Details: {str(e)}")
            print(f"--- End unexpected error for request {request_id} ---\n")
            # Clean up in case of error too
            if os.path.exists(request_dir):
                shutil.rmtree(request_dir)
            if os.path.exists(target_video_path):
                os.remove(target_video_path)
            if target_srt_path and os.path.exists(target_srt_path):
                os.remove(target_srt_path)
            return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

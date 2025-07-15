# counting_app/run_cli.py
import argparse
import os
from app.models import ObjectCounterModel
from app.main import REGION_POINTS, MODEL_PATH, OUTPUT_DIR, VIDEO_DIR # Import configs from main

def main():
    """
    Main function for the command-line interface.
    Parses arguments and runs the object counting model.
    """
    parser = argparse.ArgumentParser(description="Perform object counting on a video file.")
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="counted_output.mp4",
        help="Name for the output video file (will be saved in data/output)."
    )
    parser.add_argument(
        "--show_display",
        action="store_true",
        help="Set to True to display the video processing in real-time."
    )
    # Add more arguments for region_points, classes if you want them to be configurable via CLI

    args = parser.parse_args()

    # Ensure directories exist (redundant if main.py is run, but good for standalone CLI)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    input_path = os.path.join(VIDEO_DIR, os.path.basename(args.input_video))
    output_path = os.path.join(OUTPUT_DIR, args.output_name)

    # Copy input video to the designated videos directory if it's not already there
    if not os.path.exists(input_path):
        import shutil
        shutil.copy(args.input_video, input_path)
        print(f"Copied '{args.input_video}' to '{input_path}'")

    try:
        counter_model = ObjectCounterModel(
            model_path=MODEL_PATH,
            region_points=REGION_POINTS,
            show_display=args.show_display
        )
        success = counter_model.process_video(input_path, output_path)

        if success:
            print(f"CLI processing complete. Output saved to: {output_path}")
        else:
            print("CLI processing failed.")
    except Exception as e:
        print(f"An error occurred during CLI processing: {e}")

if __name__ == "__main__":
    main()
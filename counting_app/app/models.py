"""
This module defines the ObjectCounterModel class for performing object counting using Ultralytics YOLO.
"""

import cv2
from ultralytics import solutions, YOLO
from typing import Callable, Optional

class ObjectCounterModel:
    """
    A class to encapsulate the YOLO object counting logic.

    Attributes:
        model_path (str): Path to the YOLO model weights.
        region_points (list): Coordinates defining the counting region.
        show_display (bool): Whether to display the output during processing.
        classes (list, optional): List of specific classes to count.
    """

    def __init__(self, model_path: str, region_points: list, show_display: bool = False, classes: list = None):
        """
        Initializes the ObjectCounterModel with model path, region points, and display settings.

        Args:
            model_path (str): Path to the YOLO model weights (e.g., "yolo11n.pt").
            region_points (list): A list of tuples, where each tuple represents (x, y) coordinates
                                  defining the polygon or line for object counting.
                                  Example: [(20, 400), (1080, 400)] for a line,
                                           [(20, 400), (1080, 400), (1080, 360), (20, 360)] for a rectangle.
            show_display (bool): If True, the processed video frames will be displayed in a window.
                                 Defaults to False.
            classes (list, optional): An optional list of integers representing the specific class IDs
                                      from the COCO dataset (or your custom dataset) that you want to count.
                                      For example, [0, 2] might represent 'person' and 'car' if using a COCO-pretrained model.
                                      If None, all detected classes will be counted.
        """
        self.model_path = model_path
        self.region_points = region_points
        self.show_display = show_display
        self.classes = classes
        self.counter = self._initialize_counter()

    def _initialize_counter(self):
        """
        Initializes and returns an Ultralytics ObjectCounter instance.

        Returns:
            ultralytics.solutions.ObjectCounter: An initialized ObjectCounter object.
        """
        # Load the YOLO model
        model = YOLO(self.model_path)
        return solutions.ObjectCounter(
            model=model,
            region=self.region_points,
            show=self.show_display,
            classes=self.classes,
            line_width=2,
        )

    def process_video(self, video_path: str, output_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        """
        Processes a video file to count objects and saves the output.

        Args:
            video_path (str): The path to the input video file.
            output_path (str): The path where the output video file will be saved.
            progress_callback (Optional[Callable[[int, int], None]]): An optional callback function
                                                                       that will be called with
                                                                       (current_frame, total_frames)
                                                                       during processing.

        Returns:
            bool: True if the video was processed successfully, False otherwise.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT,
                                               cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            cap.release()
            return False

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            results = self.counter(im0)
            video_writer.write(results.plot_im)
            frame_count += 1

            # Call the progress callback if provided
            if progress_callback:
                progress_callback(frame_count, total_frames)

        cap.release()
        video_writer.release()
        print(f"Video processing complete. Output saved to: {output_path}")
        return True


import cv2
import os

class helpers:
    """
    A simple helper class to provide basic image operations.
    """
    @staticmethod
    def display(window_name, img):
        print(f"DEBUG: Attempting to display image '{window_name}'. (Display functions may not work in headless environments.)")
        # cv2.imshow(window_name, img)
        # cv2.waitKey(1)

    @staticmethod
    def save_image(filepath, img):
        try:
            dirname = os.path.dirname(filepath)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            cv2.imwrite(filepath, img)
            print(f"DEBUG: Saved image to {filepath}")
        except Exception as e:
            print(f"ERROR: Could not save image to {filepath}: {e}")
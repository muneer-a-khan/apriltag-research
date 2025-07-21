import os
import subprocess
import signal
from pyk4a import PyK4A, Config, ColorResolution
import cv2

def capture_kinect_image():
    """Captures one image from Kinect color camera"""
    k4a = PyK4A(Config(
        color_resolution=ColorResolution.RES_1080P,
        depth_mode=None
    ))

    try:
        k4a.start()
        capture = k4a.get_capture()
        if capture.color is not None:
            filename = "kinect_rgb_image.png"
            cv2.imwrite(filename, capture.color)
            print(f"Saved color image to {filename}")
        else:
            print("No color image captured.")
    finally:
        k4a.stop()

if __name__ == "__main__":
    capture_kinect_image()


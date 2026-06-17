"""
Experiment 15: Perform Edge detection using the Sobel matrix along the XY axis.

The X and Y gradients are combined into a single edge-magnitude image.

Run:  python exp19_sobel_xy.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude = sqrt(gx^2 + gy^2), then scale to 8-bit.
    magnitude = cv2.magnitude(gx, gy)
    sobel_xy = cv2.convertScaleAbs(magnitude)

    cv2.imwrite("output_sobel_xy.png", sobel_xy)
    print("Saved output_sobel_xy.png")
    show({"Original": img, "Sobel XY (magnitude)": sobel_xy})


if __name__ == "__main__":
    main()

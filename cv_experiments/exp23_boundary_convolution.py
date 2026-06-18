"""
Experiment 23: Find the boundary (edges) of an image using a convolution
kernel.

A boundary/outline can be extracted with a Laplacian-style edge kernel that
sums to zero, so flat regions become 0 (black) and only intensity changes
(boundaries) survive:

        [ -1 -1 -1 ]
        [ -1  8 -1 ]
        [ -1 -1 -1 ]

Run:  python exp23_boundary_convolution.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Boundary-detection (outline) convolution kernel.
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)

    boundary = cv2.filter2D(gray, cv2.CV_32F, kernel)
    boundary = cv2.convertScaleAbs(boundary)

    cv2.imwrite("output_boundary.png", boundary)
    print("Saved output_boundary.png")
    show({"Original": img, "Boundary (convolution)": boundary})


if __name__ == "__main__":
    main()

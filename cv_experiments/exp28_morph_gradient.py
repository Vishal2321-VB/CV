"""
Experiment 28: Morphological operation -- Morphological Gradient (OpenCV).

Morphological gradient = dilation - erosion (cv2.MORPH_GRADIENT). The result is
the outline/boundary of the foreground objects.

Run:  python exp28_morph_gradient.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    cv2.imwrite("output_morph_gradient.png", gradient)
    print("Saved output_morph_gradient.png")
    show({"Binary": binary, "Morphological gradient": gradient})


if __name__ == "__main__":
    main()

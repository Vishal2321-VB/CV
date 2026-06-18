"""
Experiment 25: Morphological operation -- Dilation (OpenCV).

Dilation grows bright (foreground) regions: a pixel becomes white if any pixel
under the structuring element is white. It fills small holes and joins broken
parts of an object.

Run:  python exp25_dilation.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    cv2.imwrite("output_dilation.png", dilated)
    print("Saved output_dilation.png")
    show({"Binary": binary, "Dilated": dilated})


if __name__ == "__main__":
    main()

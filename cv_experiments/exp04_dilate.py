"""
Experiment 1d: Read an image and dilate it using the dilate function.

Dilation grows the bright regions of an image (useful for closing small gaps).

Run:  python exp04_dilate.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()

    # Structuring element (kernel) that defines the neighbourhood.
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)

    cv2.imwrite("output_dilate.png", dilated)
    print("Saved output_dilate.png")
    show({"Original": img, "Dilated": dilated})


if __name__ == "__main__":
    main()

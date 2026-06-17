"""
Experiment 1e: Read an image and erode it using the erode function.

Erosion shrinks the bright regions of an image (useful for removing noise).

Run:  python exp05_erode.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()

    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)

    cv2.imwrite("output_erode.png", eroded)
    print("Saved output_erode.png")
    show({"Original": img, "Eroded": eroded})


if __name__ == "__main__":
    main()

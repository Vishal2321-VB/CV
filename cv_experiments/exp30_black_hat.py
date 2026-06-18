"""
Experiment 30: Morphological operation -- Black Hat (OpenCV).

Black Hat = closing - original (cv2.MORPH_BLACKHAT). It isolates small dark
details/elements that are smaller than the structuring element.

Run:  python exp30_black_hat.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    cv2.imwrite("output_black_hat.png", blackhat)
    print("Saved output_black_hat.png")
    show({"Gray": gray, "Black Hat": blackhat})


if __name__ == "__main__":
    main()

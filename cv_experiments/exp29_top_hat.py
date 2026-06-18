"""
Experiment 29: Morphological operation -- Top Hat (OpenCV).

Top Hat = original - opening (cv2.MORPH_TOPHAT). It isolates small bright
details/elements that are smaller than the structuring element.

Run:  python exp29_top_hat.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # A larger kernel makes the bright-detail extraction clearer.
    kernel = np.ones((15, 15), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    cv2.imwrite("output_top_hat.png", tophat)
    print("Saved output_top_hat.png")
    show({"Gray": gray, "Top Hat": tophat})


if __name__ == "__main__":
    main()

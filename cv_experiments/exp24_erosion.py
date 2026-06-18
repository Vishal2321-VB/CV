"""
Experiment 24: Morphological operation -- Erosion (OpenCV).

Erosion shrinks bright (foreground) regions: a pixel stays white only if every
pixel under the structuring element is white. It removes small white noise and
thins objects.

Run:  python exp24_erosion.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binary image so the morphology is clearly visible.
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)

    cv2.imwrite("output_erosion.png", eroded)
    print("Saved output_erosion.png")
    show({"Binary": binary, "Eroded": eroded})


if __name__ == "__main__":
    main()

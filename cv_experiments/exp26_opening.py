"""
Experiment 26: Morphological operation -- Opening (OpenCV).

Opening = erosion followed by dilation (cv2.MORPH_OPEN). It removes small white
specks/noise from the background while keeping the overall object shape.

Run:  python exp26_opening.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Add some white speckle noise to make "opening" visibly remove it.
    noise = (np.random.rand(*binary.shape) > 0.97).astype(np.uint8) * 255
    noisy = cv2.bitwise_or(binary, noise)

    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)

    cv2.imwrite("output_opening.png", opened)
    print("Saved output_opening.png")
    show({"Noisy binary": noisy, "Opened": opened})


if __name__ == "__main__":
    main()

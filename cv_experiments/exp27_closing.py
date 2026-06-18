"""
Experiment 27: Morphological operation -- Closing (OpenCV).

Closing = dilation followed by erosion (cv2.MORPH_CLOSE). It fills small black
holes inside the foreground while keeping the overall object shape.

Run:  python exp27_closing.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Punch small black holes so "closing" visibly fills them.
    holes = (np.random.rand(*binary.shape) > 0.97).astype(np.uint8) * 255
    holed = cv2.bitwise_and(binary, cv2.bitwise_not(holes))

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(holed, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("output_closing.png", closed)
    print("Saved output_closing.png")
    show({"Holed binary": holed, "Closed": closed})


if __name__ == "__main__":
    main()

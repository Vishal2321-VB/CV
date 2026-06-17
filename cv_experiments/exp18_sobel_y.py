"""
Experiment 14: Perform Edge detection using the Sobel matrix along the Y axis.

The Sobel-Y kernel responds to horizontal edges (intensity change in y).

Run:  python exp18_sobel_y.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # dx=0, dy=1 -> derivative along y.
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    cv2.imwrite("output_sobel_y.png", sobel_y)
    print("Saved output_sobel_y.png")
    show({"Original": img, "Sobel Y": sobel_y})


if __name__ == "__main__":
    main()

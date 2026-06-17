"""
Experiment 13: Perform Edge detection using the Sobel matrix along the X axis.

The Sobel-X kernel responds to vertical edges (intensity change in x).

Run:  python exp17_sobel_x.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # dx=1, dy=0 -> derivative along x. CV_64F keeps negative gradients.
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)     # back to displayable 8-bit

    cv2.imwrite("output_sobel_x.png", sobel_x)
    print("Saved output_sobel_x.png")
    show({"Original": img, "Sobel X": sobel_x})


if __name__ == "__main__":
    main()

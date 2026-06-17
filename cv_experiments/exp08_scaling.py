"""
Experiment 4: Scale an image to bigger and smaller sizes.

Run:  python exp08_scaling.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()

    # Scale up 2x using cubic interpolation (good for enlarging).
    bigger = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Scale down to half using area interpolation (good for shrinking).
    smaller = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imwrite("output_bigger.png", bigger)
    cv2.imwrite("output_smaller.png", smaller)
    print(f"Original {img.shape[:2]} -> bigger {bigger.shape[:2]}, smaller {smaller.shape[:2]}")
    show({"Original": img, "Bigger (2x)": bigger, "Smaller (0.5x)": smaller})


if __name__ == "__main__":
    main()

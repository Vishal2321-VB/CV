"""
Experiment 8: Perform Perspective Transformation on the image.

A perspective (homography) transform maps four source points to four
destination points and can change the apparent viewpoint ("bird's eye").

Run:  python exp12_perspective_image.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    h, w = img.shape[:2]

    # Four corners of the source -> a warped quadrilateral.
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[w * 0.0, h * 0.2],
                      [w * 1.0, h * 0.0],
                      [w * 0.8, h * 1.0],
                      [w * 0.2, h * 0.9]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    cv2.imwrite("output_perspective.png", warped)
    print("Saved output_perspective.png")
    show({"Original": img, "Perspective": warped})


if __name__ == "__main__":
    main()

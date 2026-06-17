"""
Experiment 7: Perform Affine Transformation on the image.

An affine transform keeps lines parallel. It is defined by mapping
three points in the source to three points in the destination.

Run:  python exp11_affine.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    h, w = img.shape[:2]

    # Three source points and where they should land.
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = np.float32([[w * 0.0, h * 0.33],
                      [w * 0.85, h * 0.25],
                      [w * 0.15, h * 0.7]])

    M = cv2.getAffineTransform(src, dst)
    affine = cv2.warpAffine(img, M, (w, h))

    cv2.imwrite("output_affine.png", affine)
    print("Saved output_affine.png")
    show({"Original": img, "Affine": affine})


if __name__ == "__main__":
    main()

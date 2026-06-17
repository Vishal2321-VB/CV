"""
Experiment 10: Perform transformation using a Homography matrix.

cv2.findHomography estimates the 3x3 matrix H that maps a set of source
points to destination points (here using RANSAC for robustness). The
matrix is then applied with warpPerspective.

Run:  python exp14_homography.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    h, w = img.shape[:2]

    # Corresponding point pairs (>= 4 needed to estimate a homography).
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_pts = np.float32([[w * 0.1, h * 0.15], [w * 0.9, h * 0.05],
                          [w * 0.75, h * 0.95], [w * 0.05, h * 0.85]])

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    print("Homography matrix:\n", H)

    warped = cv2.warpPerspective(img, H, (w, h))
    cv2.imwrite("output_homography.png", warped)
    print("Saved output_homography.png")
    show({"Original": img, "Homography Warp": warped})


if __name__ == "__main__":
    main()

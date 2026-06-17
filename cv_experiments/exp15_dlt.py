"""
Experiment 11: Perform transformation using Direct Linear Transformation (DLT).

DLT estimates the homography H by building a linear system A*h = 0 from
point correspondences and solving it with SVD (the solution is the
singular vector for the smallest singular value). This is the math that
cv2.findHomography wraps -- here we implement it by hand.

Run:  python exp15_dlt.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def dlt_homography(src, dst):
    """Estimate 3x3 homography from >=4 point pairs using the DLT algorithm."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, dtype=np.float64)

    # Solve A h = 0 -> h is the last row of V^T (smallest singular value).
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]          # normalise so H[2,2] == 1


def main():
    img = load_image()
    h, w = img.shape[:2]

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[w * 0.1, h * 0.15], [w * 0.9, h * 0.05],
                      [w * 0.75, h * 0.95], [w * 0.05, h * 0.85]])

    H = dlt_homography(src, dst)
    print("DLT homography matrix:\n", H)

    warped = cv2.warpPerspective(img, H, (w, h))
    cv2.imwrite("output_dlt.png", warped)
    print("Saved output_dlt.png")
    show({"Original": img, "DLT Warp": warped})


if __name__ == "__main__":
    main()

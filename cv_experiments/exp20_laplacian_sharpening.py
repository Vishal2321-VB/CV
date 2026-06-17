"""
Experiments 16, 17 & 18: Sharpen an image using Laplacian masks.

Sharpening = original + (original - blurred edges), i.e. adding a scaled
Laplacian (second derivative) back onto the image. This file demonstrates
all three masks from the lab sheet:

  16. Laplacian with NEGATIVE centre coefficient (4-neighbour):
          [ 0  1  0 ]
          [ 1 -4  1 ]
          [ 0  1  0 ]
      Sharpen by SUBTRACTING the Laplacian: g = f - laplacian(f)

  17. Laplacian extended with DIAGONAL neighbours (8-neighbour):
          [ 1  1  1 ]
          [ 1 -8  1 ]
          [ 1  1  1 ]

  18. Laplacian with POSITIVE centre coefficient -- the "composite"
      sharpening mask, which adds the original in one convolution:
          [  0 -1  0 ]
          [ -1  5 -1 ]
          [  0 -1  0 ]
      Applying this mask directly gives the sharpened result: g = f + (f - lap)

Run:  python exp20_laplacian_sharpening.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()

    # --- 16. Negative-centre Laplacian (4-neighbour) ---
    mask_neg = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]], dtype=np.float32)
    lap_neg = cv2.filter2D(img, cv2.CV_32F, mask_neg)
    # Centre is negative, so sharpen by subtracting the Laplacian.
    sharp_neg = cv2.convertScaleAbs(img.astype(np.float32) - lap_neg)

    # --- 17. Diagonal (8-neighbour) Laplacian ---
    mask_diag = np.array([[1,  1, 1],
                          [1, -8, 1],
                          [1,  1, 1]], dtype=np.float32)
    lap_diag = cv2.filter2D(img, cv2.CV_32F, mask_diag)
    sharp_diag = cv2.convertScaleAbs(img.astype(np.float32) - lap_diag)

    # --- 18. Positive-centre composite Laplacian (adds original in one pass) ---
    mask_pos = np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]], dtype=np.float32)
    sharp_pos = cv2.filter2D(img, -1, mask_pos)

    cv2.imwrite("output_sharpen_negative.png", sharp_neg)
    cv2.imwrite("output_sharpen_diagonal.png", sharp_diag)
    cv2.imwrite("output_sharpen_positive.png", sharp_pos)
    print("Saved the three sharpened images.")

    show({"Original": img,
          "16: Negative centre": sharp_neg,
          "17: Diagonal 8-neighbour": sharp_diag,
          "18: Positive centre": sharp_pos})


if __name__ == "__main__":
    main()

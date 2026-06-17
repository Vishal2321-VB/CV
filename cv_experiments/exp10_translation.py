"""
Experiment 6: Move (translate) an image from one place to another.

Translation shifts every pixel by (tx, ty) using an affine matrix:
    | 1  0  tx |
    | 0  1  ty |

Run:  python exp10_translation.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    h, w = img.shape[:2]

    tx, ty = 100, 60                      # shift right 100px, down 60px
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    moved = cv2.warpAffine(img, M, (w, h))

    cv2.imwrite("output_translation.png", moved)
    print(f"Moved image by ({tx}, {ty}).")
    show({"Original": img, "Translated": moved})


if __name__ == "__main__":
    main()

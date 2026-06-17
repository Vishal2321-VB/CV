"""
Experiment 1b: Read an image in Python and blur it using GaussianBlur.

Run:  python exp02_gaussian_blur.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()

    # Kernel size must be odd and positive. Larger kernel -> stronger blur.
    # sigmaX=0 lets OpenCV derive sigma from the kernel size.
    blurred = cv2.GaussianBlur(img, (15, 15), 0)

    cv2.imwrite("output_gaussian_blur.png", blurred)
    print("Saved output_gaussian_blur.png")
    show({"Original": img, "Gaussian Blur": blurred})


if __name__ == "__main__":
    main()

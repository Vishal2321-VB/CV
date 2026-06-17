"""
Experiment 1a: Read an image in Python and convert it to Grayscale.

Run:  python exp01_grayscale.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()

    # Convert BGR (OpenCV's default colour order) to single-channel grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("output_grayscale.png", gray)
    print("Saved output_grayscale.png")
    show({"Original": img, "Grayscale": gray})


if __name__ == "__main__":
    main()

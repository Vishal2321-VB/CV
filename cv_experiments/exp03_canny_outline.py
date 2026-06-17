"""
Experiment 1c: Read an image and show its outline using the Canny function.

Run:  python exp03_canny_outline.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Light blur first reduces noise so Canny finds cleaner edges.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Two thresholds: edges above 150 are kept, those below 50 are dropped,
    # and in-between pixels are kept only if connected to a strong edge.
    edges = cv2.Canny(gray, 50, 150)

    cv2.imwrite("output_canny.png", edges)
    print("Saved output_canny.png")
    show({"Original": img, "Canny Outline": edges})


if __name__ == "__main__":
    main()

"""
Experiment 12: Perform Edge detection using the Canny method.

Run:  python exp16_edge_canny.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 100, 200)      # lower/upper hysteresis thresholds

    cv2.imwrite("output_edge_canny.png", edges)
    print("Saved output_edge_canny.png")
    show({"Original": img, "Canny Edges": edges})


if __name__ == "__main__":
    main()

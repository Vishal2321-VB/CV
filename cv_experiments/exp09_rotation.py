"""
Experiment 5: Rotate an image clockwise and counter-clockwise.

Run:  python exp09_rotation.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def rotate(img, angle):
    """Rotate around the image centre. Positive angle = counter-clockwise."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(img, M, (w, h))


def main():
    img = load_image()

    ccw = rotate(img, 45)     # +45 deg -> counter-clockwise
    cw = rotate(img, -45)     # -45 deg -> clockwise

    cv2.imwrite("output_rotate_ccw.png", ccw)
    cv2.imwrite("output_rotate_cw.png", cw)
    print("Saved rotated images.")
    show({"Original": img, "Counter-Clockwise 45": ccw, "Clockwise 45": cw})


if __name__ == "__main__":
    main()

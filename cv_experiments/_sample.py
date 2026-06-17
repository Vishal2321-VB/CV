"""Shared helper: load an image from a path, or synthesize a colourful
sample image so every experiment runs even without an asset file.

Usage in an experiment:
    from _sample import load_image
    img = load_image()                 # uses CLI arg or a generated sample
    img = load_image("photo.jpg")      # force a specific file
"""
import sys
import numpy as np
import cv2


def make_sample(width=512, height=512):
    """Create a synthetic BGR image with shapes, gradients and text."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Diagonal colour gradient background
    for y in range(height):
        for x in range(width):
            img[y, x] = ((x * 255) // width, (y * 255) // height,
                         ((x + y) * 255) // (width + height))

    # A few solid shapes to make edges/contours obvious
    cv2.rectangle(img, (60, 60), (200, 200), (255, 255, 255), -1)
    cv2.circle(img, (360, 140), 80, (0, 0, 255), -1)
    cv2.line(img, (40, 400), (470, 300), (0, 255, 0), 8)
    cv2.putText(img, "OpenCV", (70, 470), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), 5, cv2.LINE_AA)
    return img


def load_image(path=None):
    """Return a BGR image. Order of precedence: explicit path -> CLI arg -> sample."""
    if path is None and len(sys.argv) > 1:
        path = sys.argv[1]

    if path:
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] Could not read '{path}', using a generated sample image.")
            return make_sample()
        return img

    print("[info] No image path given, using a generated sample image.")
    return make_sample()


def show(title_to_image, wait=True):
    """Display a dict of {window_title: image} and wait for a key press."""
    for title, image in title_to_image.items():
        cv2.imshow(title, image)
    if wait:
        print("Press any key on an image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

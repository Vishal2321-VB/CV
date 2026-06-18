"""
Experiment 22 (second item): Do Cropping, Copying and Pasting a region of an
image inside another image using OpenCV.

Cropping is just NumPy array slicing: img[y1:y2, x1:x2]. Copying gives an
independent array, and pasting is assigning that array into another region.

Run:  python exp22b_crop_copy_paste.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()
    h, w = img.shape[:2]

    # --- Crop a region of interest (ROI) ---
    x1, y1, x2, y2 = w // 4, h // 4, w // 2, h // 2
    crop = img[y1:y2, x1:x2].copy()      # .copy() => an independent array
    print(f"Cropped region of size {crop.shape[1]}x{crop.shape[0]}")

    # --- Paste the copied region into another image at a new location ---
    pasted = img.copy()
    ch, cw = crop.shape[:2]
    px, py = w - cw - 10, h - ch - 10     # bottom-right corner
    pasted[py:py + ch, px:px + cw] = crop

    cv2.imwrite("output_crop.png", crop)
    cv2.imwrite("output_paste.png", pasted)
    print("Saved output_crop.png and output_paste.png")

    show({"Original": img, "Cropped": crop, "Pasted copy": pasted})


if __name__ == "__main__":
    main()

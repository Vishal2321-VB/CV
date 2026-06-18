"""
Experiment 22: Insert a watermark into an image using OpenCV.

Two common watermark styles are demonstrated:
  1. A semi-transparent text overlay blended with cv2.addWeighted.
  2. A small logo/image pasted into a corner.

Run:  python exp22_watermark.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def text_watermark(img, text="OpenCV (c)", alpha=0.4):
    """Blend a translucent diagonal text watermark over the image."""
    overlay = img.copy()
    h, w = img.shape[:2]
    cv2.putText(overlay, text, (int(w * 0.1), int(h * 0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, w / 400.0, (255, 255, 255), 3,
                cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def main():
    img = load_image()
    h, w = img.shape[:2]

    # 1. Transparent text watermark.
    wm_text = text_watermark(img)

    # 2. Logo watermark: build a tiny logo and paste it bottom-right.
    logo = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.putText(logo, "LOGO", (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255), 2, cv2.LINE_AA)
    lh, lw = logo.shape[:2]
    roi = wm_text[h - lh - 10:h - 10, w - lw - 10:w - 10]
    # Add the logo onto the ROI so the background shows through dark pixels.
    blended = cv2.addWeighted(roi, 1.0, logo, 0.8, 0)
    wm_text[h - lh - 10:h - 10, w - lw - 10:w - 10] = blended

    cv2.imwrite("output_watermarked.png", wm_text)
    print("Saved output_watermarked.png")
    show({"Original": img, "Watermarked": wm_text})


if __name__ == "__main__":
    main()

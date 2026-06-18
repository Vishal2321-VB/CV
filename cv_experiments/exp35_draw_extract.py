"""
Experiment 35: Draw a rectangular shape and extract the objects inside it.

Objects (contours) are found from a binary image. For every detected object a
bounding rectangle is drawn, and each enclosed region is cropped out ("extracted")
and saved as its own image.

Run:  python exp35_draw_extract.py [optional_image_path]
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxed = img.copy()
    count = 0
    for c in contours:
        if cv2.contourArea(c) < 500:        # ignore tiny noise blobs
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract (crop) the object and save it.
        obj = img[y:y + h, x:x + w]
        fname = f"output_object_{count}.png"
        cv2.imwrite(fname, obj)
        print(f"Extracted {fname}  ({w}x{h})")
        count += 1

    cv2.imwrite("output_boxed.png", boxed)
    print(f"Drew {count} rectangle(s); saved output_boxed.png")
    show({"Rectangles drawn": boxed})


if __name__ == "__main__":
    main()

"""
Experiment 21: Perform Sharpening of an image using Gradient masking.

Gradient (first-derivative) masks respond to intensity changes. The lab sheet
shows the two Sobel gradient masks:

    Sobel-Y (horizontal edges)      Sobel-X (vertical edges)
        [ -1 -2 -1 ]                    [ -1  0  1 ]
        [  0  0  0 ]                    [ -2  0  2 ]
        [  1  2  1 ]                    [ -1  0  1 ]

The gradient magnitude |G| = sqrt(Gx^2 + Gy^2) highlights edges. Sharpening is
then done by ADDING the gradient back onto the original image so that edges
become more pronounced.

Run:  python exp21_gradient_sharpening.py [optional_image_path]
"""
import cv2
import numpy as np
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gradient masks exactly as shown on the lab sheet.
    mask_y = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)
    mask_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, mask_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, mask_y)

    # Gradient magnitude -> the edge "mask".
    grad = cv2.magnitude(gx, gy)
    grad = cv2.convertScaleAbs(grad)

    # Sharpen: add the (3-channel) gradient mask back onto the original.
    grad_bgr = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
    sharpened = cv2.addWeighted(img, 1.0, grad_bgr, 0.7, 0)

    cv2.imwrite("output_gradient_mask.png", grad)
    cv2.imwrite("output_gradient_sharpened.png", sharpened)
    print("Saved output_gradient_mask.png and output_gradient_sharpened.png")

    show({"Original": img,
          "Gradient mask": grad,
          "Sharpened": sharpened})


if __name__ == "__main__":
    main()

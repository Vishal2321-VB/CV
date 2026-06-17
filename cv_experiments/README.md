# Computer Vision with OpenCV — Lab Experiments

20 self-contained Python experiments from the *List of CV Experiments* sheet.
Each script runs on its own. If you don't pass an image/video, it generates a
synthetic sample so the script still works.

## Setup

```bash
pip install opencv-python numpy
```

## Run any experiment

```bash
cd cv_experiments
python exp01_grayscale.py                 # uses a generated sample image
python exp01_grayscale.py myphoto.jpg     # or pass your own image
python exp06_video_processing.py clip.mp4 # video experiments accept a video path
```

Image windows wait for a key press; video windows: press `q` to advance/quit.

## Mapping to the lab sheet

| File | Lab item | Description |
|------|----------|-------------|
| exp01_grayscale.py | 1a | Convert image to grayscale |
| exp02_gaussian_blur.py | 1b | Blur using GaussianBlur |
| exp03_canny_outline.py | 1c | Outline using Canny |
| exp04_dilate.py | 1d | Dilate an image |
| exp05_erode.py | 1e | Erode an image |
| exp06_video_processing.py | 2 | Play video normal / slow / fast |
| exp07_webcam_capture.py | 3 | Webcam normal / slow / fast |
| exp08_scaling.py | 4 | Scale bigger and smaller |
| exp09_rotation.py | 5 | Rotate clockwise & counter-clockwise |
| exp10_translation.py | 6 | Move image from one place to another |
| exp11_affine.py | 7 | Affine transformation |
| exp12_perspective_image.py | 8 | Perspective transform on image |
| exp13_perspective_video.py | 9 | Perspective transform on video |
| exp14_homography.py | 10 | Transform via homography matrix |
| exp15_dlt.py | 11 | Transform via Direct Linear Transformation |
| exp16_edge_canny.py | 12 | Edge detection — Canny |
| exp17_sobel_x.py | 13 | Edge detection — Sobel X |
| exp18_sobel_y.py | 14 | Edge detection — Sobel Y |
| exp19_sobel_xy.py | 15 | Edge detection — Sobel XY |
| exp20_laplacian_sharpening.py | 16, 17, 18 | Laplacian sharpening (3 masks) |

`_sample.py` is a shared helper (image loader + sample generator), not an experiment.

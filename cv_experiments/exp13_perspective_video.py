"""
Experiment 9: Perform Perspective Transformation on a video.

The same homography is applied to every frame of a video stream.

Run:  python exp13_perspective_video.py path/to/video.mp4
If no video is given, a synthetic clip is generated and used.
"""
import sys
import cv2
import numpy as np


def make_sample_video(path="sample_video.avi", frames=120, size=(480, 480)):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, 30.0, size)
    w, h = size
    for i in range(frames):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        x = int((i / frames) * w)
        cv2.circle(frame, (x, h // 2), 40, (0, 200, 255), -1)
        cv2.rectangle(frame, (50, 50), (200, 200), (255, 255, 0), 3)
        writer.write(frame)
    writer.release()
    print(f"Generated {path}")
    return path


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else make_sample_video()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[error] cannot open {path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[w * 0.0, h * 0.2], [w * 1.0, h * 0.0],
                      [w * 0.8, h * 1.0], [w * 0.2, h * 0.9]])
    M = cv2.getPerspectiveTransform(src, dst)

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        warped = cv2.warpPerspective(frame, M, (w, h))
        cv2.imshow("Original", frame)
        cv2.imshow("Perspective", warped)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

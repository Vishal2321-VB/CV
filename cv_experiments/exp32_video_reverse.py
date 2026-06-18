"""
Experiment 32: Using OpenCV, play a video in reverse mode.

All frames are read into a list, then displayed from the last frame to the
first. (For very long videos you would seek with CAP_PROP_POS_FRAMES instead of
buffering everything, but buffering is simplest for a lab demo.)

Run:  python exp32_video_reverse.py path/to/video.mp4
If no video is given, a short synthetic clip is generated and reversed.
"""
import sys
import cv2
import numpy as np


def make_sample_video(path="sample_reverse.avi", frames=90, size=(480, 480)):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, 30.0, size)
    w, h = size
    for i in range(frames):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        x = int((i / frames) * w)
        cv2.circle(frame, (x, h // 2), 40, (0, 200, 255), -1)
        cv2.putText(frame, f"frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"Read {len(frames)} frames; playing in reverse. Press 'q' to quit.")

    for frame in reversed(frames):
        cv2.imshow("Reverse playback", frame)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

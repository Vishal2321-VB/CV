"""
Experiment 2: Read a captured video file and display it normally,
in slow motion and in fast motion.

Slow/fast motion is achieved purely by changing the per-frame delay
(waitKey) -- a larger delay plays slower, a smaller delay plays faster.

Run:  python exp06_video_processing.py path/to/video.mp4
If no video is given, a short synthetic clip is generated and used.
"""
import sys
import cv2
import numpy as np


def make_sample_video(path="sample_video.avi", frames=120, size=(480, 480)):
    """Generate a small clip of a moving circle so the demo runs without assets."""
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


def play(path, delay, title):
    """Play a video with a fixed inter-frame delay (ms)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[error] cannot open {path}")
        return
    print(f"Playing '{title}' (delay={delay}ms). Press 'q' to skip.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow(title, frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyWindow(title)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else make_sample_video()
    play(path, delay=33, title="Normal Speed")     # ~30 fps
    play(path, delay=120, title="Slow Motion")      # longer delay -> slower
    play(path, delay=5, title="Fast Motion")        # shorter delay -> faster
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

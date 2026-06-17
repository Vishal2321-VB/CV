"""
Experiment 3: Capture video from the web camera and display it
normally, in slow motion and in fast motion.

Slow/fast motion here is simulated by repeating each frame (slow) or
skipping frames (fast) while displaying a live webcam feed.

Run:  python exp07_webcam_capture.py
Press 'q' to quit, 's' for slow, 'f' for fast, 'n' for normal.
"""
import cv2


def main():
    cap = cv2.VideoCapture(0)          # 0 = default web camera
    if not cap.isOpened():
        print("[error] Cannot access the web camera.")
        return

    mode = "normal"
    delay = {"normal": 30, "slow": 150, "fast": 5}

    print("Modes: 'n' normal | 's' slow | 'f' fast | 'q' quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[error] Failed to read frame.")
            break

        cv2.putText(frame, f"mode: {mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(delay[mode]) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            mode = "slow"
        elif key == ord("f"):
            mode = "fast"
        elif key == ord("n"):
            mode = "normal"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Experiment 34: Vehicle detection in a video frame using OpenCV.

Each frame is run through a Haar cascade classifier trained for cars and every
detection is boxed. OpenCV does NOT ship a car cascade, so download a
'cars.xml' Haar cascade (widely available, e.g. from the OpenCV car-detection
samples) and place it next to this script, or pass its path.

Run:
    python exp34_vehicle_detection.py traffic.mp4
    python exp34_vehicle_detection.py traffic.mp4 cars.xml
If the cascade is missing, the script falls back to the bundled full-body
cascade so the detection pipeline still runs.
"""
import os
import sys
import cv2


def get_cascade(path):
    if path and os.path.exists(path):
        return cv2.CascadeClassifier(path)
    if os.path.exists("cars.xml"):
        return cv2.CascadeClassifier("cars.xml")
    print("[warn] no car cascade found; falling back to haarcascade_fullbody.")
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_fullbody.xml")


def main():
    if len(sys.argv) < 2:
        print("Usage: python exp34_vehicle_detection.py <video> [cars.xml]")
        print("Provide a traffic video to detect vehicles.")
        return

    video_path = sys.argv[1]
    cascade_path = sys.argv[2] if len(sys.argv) > 2 else None
    cascade = get_cascade(cascade_path)
    if cascade.empty():
        print("[error] could not load a cascade classifier.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] cannot open {video_path}")
        return

    print("Detecting vehicles. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vehicles = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=3, minSize=(40, 40))
        for (x, y, w, h) in vehicles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Vehicle", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Vehicle detection", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

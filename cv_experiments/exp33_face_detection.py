"""
Experiment 33: Face detection using OpenCV.

Uses the pre-trained Haar cascade classifier that ships with OpenCV
(haarcascade_frontalface_default.xml) to detect faces and draw a rectangle
around each one.

Run:
    python exp33_face_detection.py [optional_image_path]
If no image is given (or no face is found in the sample), supply a photo of a
person, e.g.  python exp33_face_detection.py people.jpg
"""
import cv2
from _sample import load_image, show


def main():
    img = load_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[error] could not load the face cascade.")
        return

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))
    print(f"Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    cv2.imwrite("output_faces.png", img)
    print("Saved output_faces.png")
    show({"Face detection": img})


if __name__ == "__main__":
    main()

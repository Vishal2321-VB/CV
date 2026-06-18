"""
Experiment 31: Recognise an object (e.g. a watch) in an image by general object
recognition using OpenCV (ORB feature matching).

A reference/template image of the object is matched against a scene image using
ORB keypoints + a brute-force Hamming matcher. If enough good matches are found,
the object's location is outlined with a bounding box via homography.

Run:
    python exp31_object_recognition.py object.jpg scene.jpg
If no images are given, a synthetic object and scene are generated so the
matching pipeline still runs end to end.
"""
import sys
import cv2
import numpy as np


def make_object_and_scene():
    """Create a small object patch and a scene that contains it (rotated)."""
    obj = np.full((120, 120, 3), 40, dtype=np.uint8)
    cv2.circle(obj, (60, 60), 50, (200, 200, 200), 3)
    cv2.circle(obj, (60, 60), 5, (255, 255, 255), -1)
    cv2.putText(obj, "12", (48, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    scene = np.full((480, 640, 3), 70, dtype=np.uint8)
    m = cv2.getRotationMatrix2D((60, 60), 25, 1.4)
    placed = cv2.warpAffine(obj, m, (180, 180))
    scene[150:330, 220:400] = placed
    return obj, scene


def main():
    if len(sys.argv) >= 3:
        obj = cv2.imread(sys.argv[1])
        scene = cv2.imread(sys.argv[2])
        if obj is None or scene is None:
            print("[warn] could not read inputs, using generated samples.")
            obj, scene = make_object_and_scene()
    else:
        print("[info] no inputs given, using generated object & scene.")
        obj, scene = make_object_and_scene()

    orb = cv2.ORB_create(1000)
    k1, d1 = orb.detectAndCompute(obj, None)
    k2, d2 = orb.detectAndCompute(scene, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)
    good = matches[:30]
    print(f"Found {len(good)} good matches.")

    result = cv2.drawMatches(obj, k1, scene, k2, good, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Outline the recognised object using homography if we have enough matches.
    if len(good) >= 4:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            h, w = obj.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, H)
            scene_box = scene.copy()
            cv2.polylines(scene_box, [np.int32(proj)], True, (0, 255, 0), 3)
            cv2.imwrite("output_object_located.png", scene_box)
            print("Saved output_object_located.png")

    cv2.imwrite("output_object_matches.png", result)
    print("Saved output_object_matches.png")
    cv2.imshow("Object recognition (matches)", result)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

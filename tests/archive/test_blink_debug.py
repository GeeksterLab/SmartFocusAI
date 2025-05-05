# tests/test_blink.py

import cv2
from gaze.face_mesh import FaceMeshDetector
from gaze.blink_detector import BlinkDetector

def main():
    fm = FaceMeshDetector(max_num_faces=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    bd = BlinkDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    print("Clignements détectés: 0", end="", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = fm.process(frame)
        if lm:
            count = bd.update(frame, lm, fm)
            # affiche à la console dès que ça change
            print(f"\rClignements détectés: {count}", end="", flush=True)

        cv2.imshow("Test Blink Counter", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print()  # saut de ligne après sortie

if __name__ == "__main__":
    main()

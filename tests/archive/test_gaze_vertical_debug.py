# test_gaze_vertical_debug.py

import cv2
import numpy as np
from gaze.face_mesh import FaceMeshDetector
from gaze.blink_detector import BlinkDetector

def main():
    fm = FaceMeshDetector(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    # En-tête de calibration verticale
    print("\n— Calibration verticale du regard —")
    print("  Regarde en HAUT   → appuie sur 'u'")
    print("  Regarde au CENTRE → appuie sur 'c'")
    print("  Regarde en BAS    → appuie sur 'd'")
    print("  Quand tu as assez, 'q' pour quitter et afficher le résumé\n")

    samples = {'up': [], 'center': [], 'down': []}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = fm.process(frame)
        if lm:
            # Points œil / iris
            LE = BlinkDetector.LEFT_EYE_IDX
            LI = FaceMeshDetector.LEFT_IRIS_IDXS
            eye_pts = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LE], dtype=np.float32)
            iris_pts = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LI], dtype=np.float32)

            # Centres
            eye_c = eye_pts[[0,3]].mean(axis=0)
            iris_c = iris_pts.mean(axis=0)

            # Vecteur normalisé (x,y)
            vec = (iris_c - eye_c) / np.linalg.norm(eye_pts[3] - eye_pts[0])

            # Affichage
            cv2.circle(frame, tuple(eye_c.astype(int)), 5, (255,0,0), -1)
            cv2.circle(frame, tuple(iris_c.astype(int)), 5, (0,0,255), -1)
            cv2.line(frame, tuple(eye_c.astype(int)), tuple(iris_c.astype(int)), (0,255,0), 2)
            cv2.putText(
                frame,
                f"x={vec[0]:+.2f}  y={vec[1]:+.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
            )
            print(f"\rDEBUG vec[1] = {vec[1]:+.2f}", end="")

        cv2.imshow("Calibration Verticale", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('u'):
            samples['up'].append(vec[1])
            print(f"\n→ UP sample:    {vec[1]:+.2f}")
        elif key == ord('c'):
            samples['center'].append(vec[1])
            print(f"\n→ CENTER sample:{vec[1]:+.2f}")
        elif key == ord('d'):
            samples['down'].append(vec[1])
            print(f"\n→ DOWN sample:  {vec[1]:+.2f}")
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Résumé
    print("\n— Résumé calibration verticale —")
    for pos in ('up','center','down'):
        arr = samples[pos]
        if arr:
            print(f"{pos:6}: {len(arr):3} samples — min {min(arr):+.2f}, max {max(arr):+.2f}, mean {np.mean(arr):+.2f}")
        else:
            print(f"{pos:6}: aucun échantillon")

if __name__ == "__main__":
    main()
import cv2
import numpy as np
from gaze.face_mesh import FaceMeshDetector
from gaze.blink_detector import BlinkDetector

def main():
    fm = FaceMeshDetector(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    # En-tête de calibration verticale
    print("\n— Calibration verticale du regard —")
    print("  Regarde en HAUT   → appuie sur 'u'")
    print("  Regarde au CENTRE → appuie sur 'c'")
    print("  Regarde en BAS    → appuie sur 'd'")
    print("  Quand tu as assez, 'q' pour quitter et afficher le résumé\n")

    samples = {'up': [], 'center': [], 'down': []}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = fm.process(frame)
        if lm:
            # Points œil / iris
            LE = BlinkDetector.LEFT_EYE_IDX
            LI = FaceMeshDetector.LEFT_IRIS_IDXS
            eye_pts = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LE], dtype=np.float32)
            iris_pts = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LI], dtype=np.float32)

            # Centres
            eye_c = eye_pts[[0,3]].mean(axis=0)
            iris_c = iris_pts.mean(axis=0)

            # Vecteur normalisé (x,y)
            vec = (iris_c - eye_c) / np.linalg.norm(eye_pts[3] - eye_pts[0])

            # Affichage
            cv2.circle(frame, tuple(eye_c.astype(int)), 5, (255,0,0), -1)
            cv2.circle(frame, tuple(iris_c.astype(int)), 5, (0,0,255), -1)
            cv2.line(frame, tuple(eye_c.astype(int)), tuple(iris_c.astype(int)), (0,255,0), 2)
            cv2.putText(
                frame,
                f"x={vec[0]:+.2f}  y={vec[1]:+.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
            )
            print(f"\rDEBUG vec[1] = {vec[1]:+.2f}", end="")

        cv2.imshow("Calibration Verticale", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('u'):
            samples['up'].append(vec[1])
            print(f"\n→ UP sample:    {vec[1]:+.2f}")
        elif key == ord('c'):
            samples['center'].append(vec[1])
            print(f"\n→ CENTER sample:{vec[1]:+.2f}")
        elif key == ord('d'):
            samples['down'].append(vec[1])
            print(f"\n→ DOWN sample:  {vec[1]:+.2f}")
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Résumé
    print("\n— Résumé calibration verticale —")
    for pos in ('up','center','down'):
        arr = samples[pos]
        if arr:
            print(f"{pos:6}: {len(arr):3} samples — min {min(arr):+.2f}, max {max(arr):+.2f}, mean {np.mean(arr):+.2f}")
        else:
            print(f"{pos:6}: aucun échantillon")

if __name__ == "__main__":
    main()

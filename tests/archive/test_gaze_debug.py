# test_gaze_debug.py

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

    print("\n— Calibration du regard —")
    print("  Regarde à GAUCHE, appuie sur 'l' pour capter")
    print("  Regarde au CENTRE, appuie sur 'c'")
    print("  Regarde à DROITE, appuie sur 'r'")
    print("  Quand tu as assez, 'q' pour quitter et afficher le résumé\n")

    samples = {'left': [], 'center': [], 'right': []}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = fm.process(frame)
        if lm:
            LE = BlinkDetector.LEFT_EYE_IDX
            LI = FaceMeshDetector.LEFT_IRIS_IDXS

            eye_pts  = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LE], np.float32)
            iris_pts = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LI], np.float32)

            eye_c  = eye_pts[[0,3]].mean(axis=0)
            iris_c = iris_pts.mean(axis=0)

            # vecteur normalisé
            vec = (iris_c - eye_c) / np.linalg.norm(eye_pts[3] - eye_pts[0])

            # affichage DEBUG à l’écran et console
            cv2.circle(frame, tuple(eye_c.astype(int)), 5, (255,0,0), -1)
            cv2.circle(frame, tuple(iris_c.astype(int)),5, (0,0,255),-1)
            cv2.line(frame, tuple(eye_c.astype(int)), tuple(iris_c.astype(int)), (0,255,0), 2)
            cv2.putText(frame, f"vec x = {vec[0]:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            print(f"\rDEBUG vec[0] = {vec[0]:.2f}", end="")

        cv2.imshow("Gaze Debug", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('l'):
            samples['left'].append(vec[0])
            print(f"\n→ LEFT sample added: {vec[0]:.2f}")
        elif key == ord('c'):
            samples['center'].append(vec[0])
            print(f"\n→ CENTER sample added: {vec[0]:.2f}")
        elif key == ord('r'):
            samples['right'].append(vec[0])
            print(f"\n→ RIGHT sample added: {vec[0]:.2f}")
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n— Résumé calibration —")
    for d in ('left','center','right'):
        arr = samples[d]
        if arr:
            print(f"{d:6}: {len(arr):3} samples — min {min(arr):.2f},  max {max(arr):.2f}, mean {np.mean(arr):.2f}")
        else:
            print(f"{d:6}: aucun échantillon")

if __name__ == "__main__":
    main()

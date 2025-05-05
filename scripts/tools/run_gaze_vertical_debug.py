# run_gaze_vertical_debug.py

import cv2
import numpy as np
from gaze.face_mesh import FaceMeshDetector
from gaze.blink_detector import BlinkDetector
from gaze.gaze_estimator import GazeEstimator

def main():
    fm = FaceMeshDetector(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Remplace ces bornes par celles issues de test_gaze_vertical_debug.py
    calibration_ranges_v = {
        "Up":     (-float("inf"), -0.05),   # hist ≤ -0.05
        "Center": (-0.05,          -0.04), # -0.05 < hist ≤ -0.04
        "Down":   (-0.04,           float("inf")), # hist > -0.04
    }

    ge_v = GazeEstimator(calibration_ranges_v, momentum=0.8, axis=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = fm.process(frame)
        if lm:
            # calcul raw vertical, smoothing, classification
            direction = ge_v.estimate(frame, lm)

            # –– debug visuel : iris → œil
            LE = BlinkDetector.LEFT_EYE_IDX
            LI = FaceMeshDetector.LEFT_IRIS_IDXS
            eye_pts  = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LE],  np.int32)
            iris_pts = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LI], np.int32)
            eye_c   = eye_pts[[0,3]].mean(axis=0).astype(int)
            iris_c  = iris_pts.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(eye_c),  5, (255,0,0), -1)
            cv2.circle(frame, tuple(iris_c), 5, (0,0,255), -1)
            # cv2.line(frame, tuple(eye_c), tuple(iris_c), (0,255,0), 2)

            # affiche Dir: Up/Center/Down et raw,hist Y
            cv2.putText(frame,
                        f"Dir: {direction or 'None'}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Vertical Gaze Calibration", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

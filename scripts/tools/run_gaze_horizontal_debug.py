# run_gaze_horizontal_debug.py

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

    # plages recalibrées d’après tes mesures
    calibration_ranges = {
        "Right":  (-0.31,  +0.03),   # tout ≤ +0.0q<QS‡3
        "Center": (+0.03,  +0.17),   # > –0.03 < hist ≤ +0.17
        "Left":   (+0.17,  +0.25),   # hist > +0.17
}

    ge = GazeEstimator(calibration_ranges, momentum=0.8)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = fm.process(frame)
        if lm:
            direction = ge.estimate(frame, lm)

            # tracé iris→œil pour debug visuel
            LE = BlinkDetector.LEFT_EYE_IDX
            LI = FaceMeshDetector.LEFT_IRIS_IDXS
            eye_pts  = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LE], np.int32)
            iris_pts = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LI], np.int32)
            eye_c   = eye_pts[[0,3]].mean(axis=0).astype(int)
            iris_c  = iris_pts.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(eye_c),  5, (255,0,0), -1)
            cv2.circle(frame, tuple(iris_c), 5, (0,0,255), -1)
            # cv2.line(frame, tuple(eye_c), tuple(iris_c), (0,255,0), 2)

            if direction:
                cv2.putText(frame, f"Dir: {direction}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Gaze Debug", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

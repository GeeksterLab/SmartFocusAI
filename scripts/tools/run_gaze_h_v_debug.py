# run_gaze_h_v_debug.py

import cv2
import numpy as np
from gaze.face_mesh import FaceMeshDetector
from gaze.blink_detector import BlinkDetector
from gaze.gaze_estimator import GazeEstimator
from gaze.smoother import DirectionSmoother

def main():
    fm = FaceMeshDetector(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 1) Plages horizontales (axis=0)
    calibration_ranges_h = {
        "Right":  (-0.31,  +0.03),   # tout ≤ +0.0q<QS‡3
        "Center": (+0.03,  +0.17),   # > –0.03 < hist ≤ +0.17
        "Left":   (+0.17,  +0.25),   # hist > +0.17
}
    ge_h = GazeEstimator(calibration_ranges_h, momentum=0.8, axis=0)

    # 2) Plages verticales (axis=1)
    calibration_ranges_v = {
        "Up":     (-float("inf"), -0.05),   # hist ≤ -0.05
        "Center": (-0.05,          -0.04), # -0.05 < hist ≤ -0.04
        "Down":   (-0.04,           float("inf")), # hist > -0.04
    }
    ge_v = GazeEstimator(calibration_ranges_v, momentum=0.8, axis=1)

    # Soother à l'extérieur de la boucle 
    h_smoother = DirectionSmoother(window_size=5)
    v_smoother = DirectionSmoother(window_size=5)  

       # ex : "Up Right" add + si "Right + Up"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = fm.process(frame)
        if lm:
            # Estimations horizontale et verticale
            dir_h = ge_h.estimate(frame, lm) or "None"
            dir_v = ge_v.estimate(frame, lm) or "None"

            # Smoothing
            smoother_h = h_smoother.update(dir_h)
            smoother_v = v_smoother.update(dir_v)

            combo = f"{smoother_v} {smoother_h}"   

            # Trace iris → œil (mêmes points pour X et Y)
            LE = BlinkDetector.LEFT_EYE_IDX
            LI = FaceMeshDetector.LEFT_IRIS_IDXS
            eye_pts  = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LE],  np.int32)
            iris_pts = np.array([fm.landmark_to_pixel(frame, lm.landmark[i]) for i in LI], np.int32)
            eye_c   = eye_pts[[0,3]].mean(axis=0).astype(int)
            iris_c  = iris_pts.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(eye_c),  5, (255,0,0), -1)
            cv2.circle(frame, tuple(iris_c), 5, (0,0,255), -1)


            # combo = f"{dir_v} {dir_h}"       # ex : "Up Right" add + si "Right + Up"
            cv2.putText(
                frame,
                f"Dir: {combo}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0,255,0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Gaze H+V Debug", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

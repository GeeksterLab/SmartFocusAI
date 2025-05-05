# src/gaze/combined_gaze.py

import numpy as np
from src.gaze.face_mesh import FaceMeshDetector
from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.smoother import DirectionSmoother

class CombinedGazeDetector:
    """
    Combine two GazeEstimators (horizontal + vertical) + smoothing to
    determine if the user is looking at the center of the screen.
    """
    def __init__(self, face_mesh: FaceMeshDetector):
        self.fm = face_mesh

        # Horizontal calibration ranges
        cal_h = {
            "Right":  (-0.31, +0.03),
            "Center": (+0.03, +0.17),
            "Left":   (+0.17, +0.25),
        }
        self.ge_h = GazeEstimator(calibration_ranges=cal_h,
                                  momentum=0.8, axis=0)

        # Vertical calibration ranges
        cal_v = {
            "Up":     (-float("inf"), -0.05),
            "Center": (-0.05, -0.04),
            "Down":   (-0.04, float("inf")),
        }
        self.ge_v = GazeEstimator(calibration_ranges=cal_v,
                                  momentum=0.8, axis=1)

        # Smoothers for each axis
        self.h_smoother = DirectionSmoother(window_size=5)
        self.v_smoother = DirectionSmoother(window_size=5)

    def is_gazing(self, frame) -> bool:
        lm = self.fm.process(frame)
        if lm is None:
            return False

        # --- estimation & smoothing ---
        dir_h      = self.ge_h.estimate(frame, lm) or "None"
        dir_v      = self.ge_v.estimate(frame, lm) or "None"
        smoother_h = self.h_smoother.update(dir_h)
        smoother_v = self.v_smoother.update(dir_v)

        # Focus if either axis is centered
        return (smoother_h == "Center") or (smoother_v == "Center")

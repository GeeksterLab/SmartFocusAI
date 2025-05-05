# # src/gaze/gaze_estimator.py

# import cv2
# import numpy as np
# from src.gaze.face_mesh import FaceMeshDetector
# from src.gaze.blink_detector import BlinkDetector

# class GazeEstimator:
#     LEFT_IRIS_IDXS  = FaceMeshDetector.LEFT_IRIS_IDXS
#     RIGHT_IRIS_IDXS = FaceMeshDetector.RIGHT_IRIS_IDXS

#     def __init__(
#         self,
#         calibration_ranges: dict | None = None,
#         *,
#         momentum: float = 0.8,
#         axis: int = 0,
#         thresh: float | None = None
#     ):
#         """
#         Deux APIs :
#          - nouvelle : calibration_ranges=dict(direction->(min,max))
#          - legacy  : un unique seuil `thresh` pour n'interpréter que la zone "Center"
#         """
#         if calibration_ranges is None:
#             if thresh is None:
#                 raise ValueError("GazeEstimator : il faut calibration_ranges **ou** thresh")
#             calibration_ranges = {"Center": (-thresh, thresh)}

#         self.calibration_ranges = calibration_ranges
#         self.momentum = momentum
#         self.axis     = axis
#         self._hist    = 0.0
#         self.fm       = FaceMeshDetector()

#     @staticmethod
#     def _to_px(frame, lm):
#         h, w = frame.shape[:2]
#         return np.array((lm.x * w, lm.y * h), dtype=np.float32)

#     def estimate(self, frame, face_landmarks):
#         lm = getattr(face_landmarks, "landmark", None)
#         if lm is None:
#             return None

#         # 1) Eye landmarks
#         LE, RE = BlinkDetector.LEFT_EYE_IDX, BlinkDetector.RIGHT_EYE_IDX
#         eye_pts_L = np.array([self._to_px(frame, lm[i]) for i in LE], np.float32)
#         eye_pts_R = np.array([self._to_px(frame, lm[i]) for i in RE], np.float32)
#         eye_c_L, eye_c_R = eye_pts_L[[0,3]].mean(0), eye_pts_R[[0,3]].mean(0)

#         # 2) Iris landmarks
#         iris_pts_L = np.array([self._to_px(frame, lm[i]) for i in self.LEFT_IRIS_IDXS], np.float32)
#         iris_pts_R = np.array([self._to_px(frame, lm[i]) for i in self.RIGHT_IRIS_IDXS], np.float32)
#         iris_c_L, iris_c_R = iris_pts_L.mean(0), iris_pts_R.mean(0)

#         # 3) Vecteurs normalisés
#         vec_L = (iris_c_L - eye_c_L) / np.linalg.norm(eye_pts_L[3] - eye_pts_L[0])
#         vec_R = (iris_c_R - eye_c_R) / np.linalg.norm(eye_pts_R[3] - eye_pts_R[0])
#         raw   = float((vec_L[self.axis] + vec_R[self.axis]) / 2.0)

#         # 4) Smoothing
#         self._hist = self._hist * self.momentum + raw * (1.0 - self.momentum)

#         # 5) Classification
#         candidates = [d for d,(mn,mx) in self.calibration_ranges.items()
#                       if mn <= self._hist <= mx]
#         if not candidates:
#             return None
#         if len(candidates) == 1:
#             return candidates[0]
#         centers = {d:(mn+mx)/2 for d,(mn,mx) in self.calibration_ranges.items()}
#         return min(candidates, key=lambda d: abs(self._hist - centers[d]))
    
#     def is_gazing(self, frame) -> bool:
#         """
#         API utilisée par FocusManager et par les stubs d’intégration.
#         Renvoie True si la direction estimée est "Center".
#         """
#         lm = self.fm.process(frame)
#         if lm is None:
#             return False
#         return (self.estimate(frame, lm) == "Center")


# src/gaze/gaze_estimator.py

import cv2
import numpy as np
from src.gaze.face_mesh import FaceMeshDetector
from src.gaze.blink_detector import BlinkDetector

class GazeEstimator:
    LEFT_IRIS_IDXS  = FaceMeshDetector.LEFT_IRIS_IDXS
    RIGHT_IRIS_IDXS = FaceMeshDetector.RIGHT_IRIS_IDXS

    def __init__(
        self,
        calibration_ranges: dict | None = None,
        *,
        momentum: float = 0.8,
        axis: int = 0,
        thresh: float | None = None
    ):
        """
        Two APIs :
         - new     : calibration_ranges=dict(direction->(min,max))
         - legacy  : a single threshold `thresh` to interpret only the "Center" zone
        """
        if calibration_ranges is None:
            if thresh is None:
                raise ValueError("GazeEstimator: must provide calibration_ranges **or** thresh")
            calibration_ranges = {"Center": (-thresh, thresh)}

        self.calibration_ranges = calibration_ranges
        self.momentum = momentum
        self.axis     = axis
        self._hist    = 0.0
        self.fm       = FaceMeshDetector()

    @staticmethod
    def _to_px(frame, lm):
        h, w = frame.shape[:2]
        return np.array((lm.x * w, lm.y * h), dtype=np.float32)

    def estimate(self, frame, face_landmarks):
        lm = getattr(face_landmarks, "landmark", None)
        if lm is None:
            return None

        # 1) Eye landmarks
        LE, RE = BlinkDetector.LEFT_EYE_IDX, BlinkDetector.RIGHT_EYE_IDX
        eye_pts_L = np.array([self._to_px(frame, lm[i]) for i in LE], np.float32)
        eye_pts_R = np.array([self._to_px(frame, lm[i]) for i in RE], np.float32)
        eye_c_L, eye_c_R = eye_pts_L[[0,3]].mean(0), eye_pts_R[[0,3]].mean(0)

        # 2) Iris landmarks
        iris_pts_L = np.array([self._to_px(frame, lm[i]) for i in self.LEFT_IRIS_IDXS], np.float32)
        iris_pts_R = np.array([self._to_px(frame, lm[i]) for i in self.RIGHT_IRIS_IDXS], np.float32)
        iris_c_L, iris_c_R = iris_pts_L.mean(0), iris_pts_R.mean(0)

        # 3) Normalized vectors
        vec_L = (iris_c_L - eye_c_L) / np.linalg.norm(eye_pts_L[3] - eye_pts_L[0])
        vec_R = (iris_c_R - eye_c_R) / np.linalg.norm(eye_pts_R[3] - eye_pts_R[0])
        raw   = float((vec_L[self.axis] + vec_R[self.axis]) / 2.0)

        # 4) Smoothing
        self._hist = self._hist * self.momentum + raw * (1.0 - self.momentum)

        # 5) Classification
        candidates = [d for d,(mn,mx) in self.calibration_ranges.items()
                      if mn <= self._hist <= mx]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        centers = {d:(mn+mx)/2 for d,(mn,mx) in self.calibration_ranges.items()}
        return min(candidates, key=lambda d: abs(self._hist - centers[d]))
    
    def is_gazing(self, frame) -> bool:
        """
        API used by FocusManager and integration stubs.
        Returns True if the estimated direction is "Center".
        """
        lm = self.fm.process(frame)
        if lm is None:
            return False
        return (self.estimate(frame, lm) == "Center")

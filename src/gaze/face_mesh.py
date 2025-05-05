# face_mesh.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Optional

class FaceMeshDetector:
    """
    Detects facial landmarks and iris points using MediaPipe Face Mesh.
    """
    LEFT_IRIS_IDXS  = [474, 475, 476, 477]
    RIGHT_IRIS_IDXS = [469, 470, 471, 472]

    def __init__(self,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame: "np.ndarray"):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        return (results.multi_face_landmarks[0]
                if results.multi_face_landmarks else None)

    @staticmethod
    def landmark_to_pixel(frame: "np.ndarray", lm) -> Tuple[int, int]:
        h, w, _ = frame.shape
        return int(lm.x * w), int(lm.y * h)

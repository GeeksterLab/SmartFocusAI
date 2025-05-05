# src/gaze/blink_detector.py

import numpy as np

class BlinkDetector:
    # Mediapipe indices for the left eye (same for the right eye if needed)
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144] 
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
    EAR_THRESH = 0.2     # threshold to detect eye closure, adjust based on your tests
    CONSEC_FRAMES = 2    # number of consecutive frames the eye must be closed
                        
    def __init__(self):
        self.counter = 0        # counts consecutive closed-eye frames
        self.blink_count = 0    # total number of blinks detected
        self.eye_closed = False

    @staticmethod
    def eye_aspect_ratio(pts):
        # pts: array of shape (6,2)
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C)

    def update(self, frame, landmarks, fm):
        """
        Call on each frame with:
          - frame: the image
          - landmarks: result of fm.process()
          - fm: instance of FaceMeshDetector to convert landmarks to pixel coordinates
        Returns the current blink_count.
        """
        # retrieve the 6 points of the left eye
        pts = np.array([
            fm.landmark_to_pixel(frame, landmarks.landmark[i])
            for i in self.LEFT_EYE_IDX
        ], dtype=np.float32)

        ear = self.eye_aspect_ratio(pts)

        if ear < self.EAR_THRESH:
            self.counter += 1
        else:
            # if the eye was closed long enough then opens, count a blink
            if self.counter >= self.CONSEC_FRAMES:
                self.blink_count += 1
            self.counter = 0

        return self.blink_count

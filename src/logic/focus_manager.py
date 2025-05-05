# src/logic/focus_manager.py

from src.detection.typing_activity import TypingActivityDetector
from src.gaze.gaze_estimator import GazeEstimator

class FocusManager:
    """
    Decides if the user is Focused or Distracted based on:
      1) If typing on the keyboard → Focused (override regardless of gaze)
      2) Otherwise, if gaze is centered → Focused
      3) Otherwise → Distracted
    """

    def __init__(self,
                 typing_detector: TypingActivityDetector,
                 gaze_detector: GazeEstimator):
        self.typing = typing_detector
        self.gaze   = gaze_detector

    def is_focused(self, frame) -> bool:
        # 1) keyboard has priority
        is_typing = self.typing.is_typing()
        if is_typing:
            # even if gaze is off-center, we remain focused
            return True

        # 2) if not typing, check gaze
        is_gazing = self.gaze.is_gazing(frame)
        if is_gazing:
            # centered gaze AND no typing → focused
            return True

        # 3) neither typing nor centered gaze → distracted
        return False

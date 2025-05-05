# tests/unit/test_blink_activity.py

import pytest
import time
import numpy as np

import src.detection.typing_activity as ta_module
from src.detection.typing_activity import TypingActivityDetector
from src.gaze.blink_detector import BlinkDetector

def test_typing_detects_key_press():
    det = TypingActivityDetector(display_timeout=1.0)
    det._on_key_press(key=None)
    assert det.is_typing()

def test_typing_after_timeout():
    det = TypingActivityDetector(display_timeout=0.01)
    time.sleep(0.02)
    assert not det.is_typing()

def test_stop_listener(monkeypatch):
    stopped = {"called": False}
    class FakeListener:
        def __init__(self, on_press): pass
        def start(self): pass
        def stop(self): stopped["called"] = True

    monkeypatch.setattr(ta_module.keyboard, "Listener",
                        lambda on_press: FakeListener(on_press))
    det = TypingActivityDetector(display_timeout=1.0)
    det.start()
    det.stop()
    assert stopped["called"]

class DummyFM:
    # simulate landmark_to_pixel just returning the point coordinates
    def landmark_to_pixel(self, frame, lm):
        # lm is a tuple (x,y)
        return np.array(lm, dtype=np.float32)

class DummyLandmarks:
    def __init__(self, pts):
        # pts is list of 6 (x,y) tuples for eye points
        # create a list of dummy landmarks that's long enough for MediaPipe indices
        # maximum index needed is 387 (from RIGHT_EYE_IDX)
        self.landmark = [(0.0, 0.0)] * 388  # create list with dummy points
        
        # map the 6 provided points to the LEFT_EYE_IDX positions used by the detector
        left_eye_idx = BlinkDetector.LEFT_EYE_IDX
        for i in range(6):
            self.landmark[left_eye_idx[i]] = pts[i]

def make_eye_points(ear_value):
    """
    Build 6 points so that EAR = ear_value:
    EAR = (A+B)/(2*C)  => choose A=B=ear_value, C=1
    So we want:
      norm(p1-p5) = ear_value
      norm(p2-p4) = ear_value
      norm(p0-p3) = 1
    We'll place:
      p0=(0,0); p3=(1,0)
      p1=(0,ear_value); p5=(0,0)
      p2=(1,ear_value); p4=(1,0)
    """
    ear = ear_value
    return [
        (0.0, 0.0),      # p0
        (0.0, ear),      # p1
        (1.0, ear),      # p2
        (1.0, 0.0),      # p3
        (1.0, 0.0),      # p4 (same as p3)
        (0.0, 0.0),      # p5 (same as p0)
    ]

def test_eye_aspect_ratio():
    pts = np.array(make_eye_points(0.5), np.float32)
    # A = norm(p1-p5)=0.5, B=norm(p2-p4)=0.5, C=norm(p0-p3)=1
    ear = BlinkDetector.eye_aspect_ratio(pts)
    assert pytest.approx(ear, rel=1e-3) == 0.5

def test_update_no_blink(monkeypatch):
    bd = BlinkDetector()
    frame = np.zeros((5,5,3))
    # build landmarks that give EAR above threshold
    pts = make_eye_points(BlinkDetector.EAR_THRESH + 0.1)
    landmarks = DummyLandmarks(pts)
    # call update twice: counter should stay 0, blink_count stays 0
    assert bd.update(frame, landmarks, DummyFM()) == 0
    assert bd.counter == 0
    assert bd.update(frame, landmarks, DummyFM()) == 0
    assert bd.counter == 0

def test_update_detects_blink(monkeypatch):
    bd = BlinkDetector()
    frame = np.zeros((5,5,3))
    # build landmarks that give EAR below threshold
    pts_closed = make_eye_points(BlinkDetector.EAR_THRESH - 0.1)
    landmarks_closed = DummyLandmarks(pts_closed)
    # simulate two closed frames
    bd.update(frame, landmarks_closed, DummyFM())
    bd.update(frame, landmarks_closed, DummyFM())
    # now open eye above threshold to trigger blink_count increment
    pts_open = make_eye_points(BlinkDetector.EAR_THRESH + 0.1)
    landmarks_open = DummyLandmarks(pts_open)
    count = bd.update(frame, landmarks_open, DummyFM())
    assert count == 1
    assert bd.counter == 0
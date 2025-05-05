# tests/unit/test_gaze_estimator.py

import numpy as np
import pytest

from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.blink_detector import BlinkDetector

class DummyLM:
    def __init__(self, x=0.0, y=0.0): self.x, self.y = x, y

@pytest.fixture(autouse=True)
def stub_indices_and_norm(monkeypatch):
    # Avoid IndexError and division by zero
    monkeypatch.setattr(BlinkDetector, "LEFT_EYE_IDX",  [0,1,2,3])
    monkeypatch.setattr(BlinkDetector, "RIGHT_EYE_IDX", [0,1,2,3])
    monkeypatch.setattr(GazeEstimator, "LEFT_IRIS_IDXS",  [0,1,2,3])
    monkeypatch.setattr(GazeEstimator, "RIGHT_IRIS_IDXS", [0,1,2,3])
    # Normalize eye landmark distance to 1
    monkeypatch.setattr(np.linalg, "norm", lambda v: 1.0)

def test_estimate_none_on_no_landmarks():
    ge = GazeEstimator(thresh=0.5, axis=0)
    class FL: landmark = None
    assert ge.estimate(None, FL) is None

def test_multiple_candidate_classification():
    ge = GazeEstimator(calibration_ranges={"A":(-1,1), "B":(-2,2)},
                      axis=0, momentum=0.0)
    FL = type("FL", (), {"landmark": [DummyLM() for _ in range(4)]})
    gaze = ge.estimate(np.zeros((5,5,3)), FL)
    assert gaze in ("A", "B")

def test_closest_center_in_overlap():
    cal = {"A":(-1,1), "X":(-0.2,0.2), "B":(-2,2)}
    ge = GazeEstimator(calibration_ranges=cal, axis=0, momentum=0.0)
    FL = type("FL", (), {"landmark": [DummyLM() for _ in range(4)]})
    gaze = ge.estimate(np.zeros((5,5,3)), FL)
    # A et X couvrent ; A est la première clé => "A"
    assert gaze == "A"

def test_smoothing_and_momentum(monkeypatch):
    ge = GazeEstimator(thresh=1.0, axis=0, momentum=0.5)
    FL = type("FL", (), {"landmark": [DummyLM() for _ in range(4)]})
    frame = np.zeros((5,5,3))
    ge.estimate(frame, FL)  # hist = 0
    # Stub _to_px pour forcer raw = +2
    monkeypatch.setattr(GazeEstimator, "_to_px",
                        lambda self, f, lm: np.array((2.0,0.0)))
    assert ge.estimate(frame, FL) == "Center"

def test_init_requires_one_of_calibration_or_thresh():
    with pytest.raises(ValueError):
        GazeEstimator()

def test_legacy_thresh_works():
    # legacy mode with thresh only → only "Center" range
    ge = GazeEstimator(thresh=0.2, axis=0, momentum=0.0)
    # stub face_landmarks.landmark to a list of DummyLM giving raw=0
    class FL: landmark = [DummyLM() for _ in range(4)]
    frame = np.zeros((5,5,3))
    # raw=0 in (-0.2,0.2) → "Center"
    assert ge.estimate(frame, FL) == "Center"

def test_no_candidates_returns_none():
    # define ranges that exclude raw=0
    ge = GazeEstimator(calibration_ranges={"A":(0.1,1.0)}, axis=0, momentum=0.0)
    class FL: landmark = [DummyLM() for _ in range(4)]
    frame = np.zeros((5,5,3))
    assert ge.estimate(frame, FL) is None

def test_multiple_candidates_closest_center():
    # two overlapping ranges, both centers at 0 -> "A" (first key) wins
    cal = {"A":(-1.0,1.0), "B":(-0.1,0.1)}
    ge = GazeEstimator(calibration_ranges=cal, axis=0, momentum=0.0)
    class FL: landmark = [DummyLM() for _ in range(4)]
    frame = np.zeros((5,5,3))
    assert ge.estimate(frame, FL) == "A"

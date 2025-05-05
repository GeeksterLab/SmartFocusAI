# tests/unit/test_combined_gaze.py

import pytest
import src.gaze.combined_gaze as cg
from src.gaze.combined_gaze import CombinedGazeDetector


class DummyFM:
    def process(self, frame):
        # Return a fake landmarks non-None
        return object()
    
# Fake GazeEstimator to return  "Center"
class FakeGE:
    def __init__(self, *a, **k): pass
    def estimate(self, frame, lm): return "Center"

# Fake GazeEstimator to return "Left"
class FakeGE_Left(FakeGE):
    def estimate(self, frame, lm): return "Left"

def test_combined_gaze_gazing(monkeypatch):
    monkeypatch.setattr(cg, "GazeEstimator", FakeGE)
    det = CombinedGazeDetector(face_mesh=DummyFM())
    # with one axis estimating "Center", should return True
    assert det.is_gazing(None) is True

def test_combined_gaze_not_center(monkeypatch):
    monkeypatch.setattr(cg, "GazeEstimator", FakeGE_Left)
    det = CombinedGazeDetector(face_mesh=DummyFM())
    # with one axis estimating "Left", should return False
    assert det.is_gazing(None) is False

def test_combined_gaze_process_none():
    class FMNone:
        def process(self, frame): return None
    det = CombinedGazeDetector(face_mesh=FMNone())
    # No face detected -> False
    assert det.is_gazing(None) is False


def test_combined_gaze_two_axes_switch():
    # Standard instance
    det = CombinedGazeDetector(face_mesh=DummyFM())

    # Stub ge_h and ge_v to return values in sequence
    seq_h = iter(["Left", "Center"])
    seq_v = iter(["None",  "None"])
    det.ge_h = type("GH", (), {"estimate": lambda self, f, lm: next(seq_h)})()
    det.ge_v = type("GV", (), {"estimate": lambda self, f, lm: next(seq_v)})()

    # Stub smoothers to return exactly the received value
    det.h_smoother = type("SH", (), {"update": lambda self, d: d})()
    det.v_smoother = type("SV", (), {"update": lambda self, d: d})()

    # 1st call: ge_h="Left", ge_v="None" -> neither axis "Center" -> False
    assert det.is_gazing(None) is False

    # 2nd call: ge_h="Center", ge_v="None" -> h_smoother="Center" -> True
    assert det.is_gazing(None) is True
# tests/unit/test_gaze_estimator_extra.py

import numpy as np
import pytest

from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.blink_detector import BlinkDetector

def test_is_gazing_api(monkeypatch):
    # Legacy mode (thresh) to simplify
    ge = GazeEstimator(thresh=0.1, axis=0)

    # No face detected -> False
    monkeypatch.setattr(ge.fm, "process", lambda frame: None)
    assert ge.is_gazing(None) is False

    # Face detected but estimate != "Center" -> False
    class FL: pass
    monkeypatch.setattr(ge.fm, "process", lambda frame: FL())
    monkeypatch.setattr(ge, "estimate", lambda frame, lm: "Left")
    assert ge.is_gazing(None) is False

    # Face + estimate "Center" -> True
    monkeypatch.setattr(ge, "estimate", lambda frame, lm: "Center")
    assert ge.is_gazing(None) is True

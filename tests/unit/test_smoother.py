# tests/unit/test_smoother.py

import pytest
from src.gaze.smoother import DirectionSmoother

def test_initial_state():
    sm = DirectionSmoother(window_size=3)
    assert sm.current == "None"

def test_update_needs_majority():
    sm = DirectionSmoother(window_size=3)
    # 1st update -> single input is enough to switch to "Left"
    assert sm.update("Left") == "Left"
    assert sm.current == "Left"
    # another direction won't switch until it is in the majority
    assert sm.update("Right") == "Left"
    assert sm.update("Right") == "Right"

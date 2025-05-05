# tests/unit/test_focus_manager.py

import pytest
from src.logic.focus_manager import FocusManager

class FakeDetector:
    def __init__(self, *a, **k): pass
    def is_typing(self): return False
    def is_gazing(self, frame): return True

def test_focus_manager_switch():
    td = FakeDetector()
    gd = FakeDetector()
    mgr = FocusManager(typing_detector=td, gaze_detector=gd)
    assert mgr.is_focused(None) is True

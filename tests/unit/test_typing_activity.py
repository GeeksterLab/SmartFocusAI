# tests/unit/test_typing_activity.py

import pytest
import src.detection.typing_activity as ta_module
from src.detection.typing_activity import TypingActivityDetector

def test_typing_detects_key_press():
    det = TypingActivityDetector(display_timeout=1.0)
    det._on_key_press(key=None)
    assert det.is_typing()

def test_typing_after_timeout():
    import time
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

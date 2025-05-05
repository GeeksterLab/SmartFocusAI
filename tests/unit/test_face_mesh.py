# tests/unit/test_face_mesh.py

import numpy as np
import pytest
import cv2

from src.gaze.face_mesh import FaceMeshDetector
import src.gaze.face_mesh as fm_module


# Global stub for cv2.cvtColor to avoid dtype errors
@pytest.fixture(autouse=True)
def stub_cvtcolor(monkeypatch):
    monkeypatch.setattr(cv2, "cvtColor", lambda img, flag: img.astype(np.uint8))

# Default stub for MediaPipe FaceMesh (no face detected)
class DummyFM:
    def __init__(self, **kw): pass
    def process(self, frame):
        class R: multi_face_landmarks = []
        return R()


@pytest.fixture(autouse=True)
def patch_mediapipe(monkeypatch):
    fake = type("M", (), {
        "solutions": type("S", (), {
            "face_mesh": type("FMMod", (), {
                "FaceMesh": lambda *a, **k: DummyFM()
            })
        })
    })()
    monkeypatch.setattr(fm_module, "mp", fake)

def test_process_no_face():
    det = FaceMeshDetector()
    assert det.process(np.zeros((10,10,3))) is None

def test_process_with_face(monkeypatch):
    lm = object()
    class DummyFM2:
        def __init__(self, **kw): pass
        def process(self, frame):
            class R: multi_face_landmarks = [lm]
            return R()
    # We just patch FaceMesh for this test
    fake2 = type("M2", (), {
        "solutions": type("S2", (), {
            "face_mesh": type("FMMod2", (), {
                "FaceMesh": lambda *a, **k: DummyFM2()
            })
        })
    })()
    monkeypatch.setattr(fm_module, "mp", fake2)

    det = FaceMeshDetector()
    res = det.process(np.zeros((10,10,3), dtype=np.uint8))
    assert res is lm


def test_process_multiple_faces(monkeypatch):

    # Stub process() returns two faces
    class DummyFM3:
        def __init__(self, **kw): pass
        def process(self, frame):
            # we retrieve two fake faces
            class Face:
                landmark = [type("P", (), {"x":0.5,"y":0.5})() for _ in range(5)]
            class R:
                multi_face_landmarks = [Face(), Face()]
            return R()   # return instance R()

    # Monkeypatch module
    fake3 = type("M3", (), {
        "solutions": type("S3", (), {
            "face_mesh": type("FMMod3", (), {
                "FaceMesh": lambda *a, **k: DummyFM3()
            })
        })
    })()
    monkeypatch.setattr(fm_module, "mp", fake3)

    # Run
    det = FaceMeshDetector()
    res = det.process(np.zeros((10,10,3), dtype=np.uint8))

    # Assert : we well receive the first face
    assert hasattr(res, "landmark")

def test_landmark_to_pixel_exact():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    # fake landmark with normalized coords
    class LM: x=0.25; y=0.5
    px, py = FaceMeshDetector.landmark_to_pixel(frame, LM)
    # x * width = 0.25*200 =50, y*height=0.5*100=50
    assert (px, py) == (50, 50)
# tests/integration/helpers.py

class OfflineTypingDetector:
    """
    Stub: deduce is_typing() from the fixture name.
    """
    def __init__(self, fixture_name: str):
        # "typing" in the name → always True
        self._typing = (fixture_name == "typing")

    def is_typing(self) -> bool:
        return self._typing


class OfflineGazeDetector:
    """
    Stub: returns True only if "center" is in the fixture name.
    """
    def __init__(self, fixture_name: str):
        self._gazing = fixture_name is not None and "center" in fixture_name

    def is_gazing(self, frame) -> bool:
        return self._gazing

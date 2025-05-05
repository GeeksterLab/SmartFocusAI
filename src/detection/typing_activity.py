# src/detection/typing_activity.py

import time
import threading

# Try real pynput; if missing (CI, cloud, etc.), provide a dummy Listener
try:
    from pynput import keyboard
except ImportError:
    class _DummyListener:
        def __init__(self, on_press):
            self._on_press = on_press
        def start(self):
            pass
        def stop(self):
            pass

    keyboard = type("keyboard", (), {"Listener": _DummyListener})


class TypingActivityDetector:
    """
    Detects keyboard typing activity by listening
    to system events via pynput (or a dummy stub if unavailable).
    """

    def __init__(self, display_timeout: float = 1.0):
        """
        :param display_timeout: duration (in seconds) during which
                                typing is considered after a key press
        """
        self.display_timeout = display_timeout
        self._last_event = 0.0
        self._lock = threading.Lock()

        # Non-blocking listener (real or dummy)
        self.listener = keyboard.Listener(on_press=self._on_key_press)

    def _on_key_press(self, key):
        """Callback called on every key press."""
        with self._lock:
            self._last_event = time.time()

    def start(self):
        """Starts listening in the background (no-op for dummy)."""
        self.listener.start()

    def stop(self):
        """Stops listening (no-op for dummy)."""
        self.listener.stop()

    def is_typing(self) -> bool:
        """
        True if a key was pressed less than display_timeout seconds ago.
        Always False if running with the dummy Listener.
        """
        with self._lock:
            return (time.time() - self._last_event) < self.display_timeout

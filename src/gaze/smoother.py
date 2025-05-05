# src/gaze/smoother.py

from collections import deque

class DirectionSmoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
        self.current = "None"

    def update(self, new_dir):
        self.window.append(new_dir)
        # if the new direction dominates more than half of the window
        if self.window.count(new_dir) > len(self.window) // 2:
            self.current = new_dir
        return self.current

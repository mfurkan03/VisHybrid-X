import time
from collections import deque


class FPSCounter:
    def __init__(self, window: int = 60):
        self.window = window
        self._times = deque(maxlen=window)
        self._start = None
        self.total_steps = 0
        self.total_time  = 0.0

    def tick(self):
        now = time.perf_counter()
        if self._start is not None:
            delta = now - self._start
            self._times.append(delta)
            self.total_time  += delta
            self.total_steps += 1
        self._start = now

    @property
    def instant_fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    @property
    def average_fps(self) -> float:
        if self.total_steps == 0 or self.total_time == 0:
            return 0.0
        return self.total_steps / self.total_time

    def summary(self) -> str:
        return (
            f"  Total Steps : {self.total_steps}\n"
            f"  Total Time  : {self.total_time:.2f} s\n"
            f"  Average FPS : {self.average_fps:.1f}\n"
            f"  Instant FPS : {self.instant_fps:.1f}"
        )
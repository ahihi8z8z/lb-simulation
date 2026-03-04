"""Traffic generation for bursty and trace-replay arrivals."""

import csv
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import simpy

from .models import Request


class TrafficGenerator:
    """Generate requests using gamma windows or trace replay timestamps."""

    def __init__(
        self,
        env: simpy.Environment,
        t_end: float,
        arrival_mode: str,
        on_request: Callable[[Request], None],
        rng: Optional[random.Random] = None,
        service_classes: int = 1,
        zipf_s: float = 1.20,
        zipf_xmin: int = 16,
        zipf_max: int = 2048,
        gamma_windows: Optional[List[Tuple[float, float, float]]] = None,
        trace_timestamps: Optional[List[float]] = None,
    ) -> None:
        self.env = env
        self.t_end = t_end
        self.arrival_mode = arrival_mode
        self.on_request = on_request
        self.rng = rng or random.Random()
        self.service_classes = max(1, service_classes)

        self.zipf_s = zipf_s
        self.zipf_xmin = zipf_xmin
        self.zipf_max = zipf_max

        self.gamma_windows = gamma_windows or []
        self.trace_timestamps = trace_timestamps or []
        self._rid = 0

    def _sample_hidden_size(self) -> int:
        # Inverse transform sampling for a truncated discrete Zipf-like distribution.
        u = self.rng.random()
        x = int(self.zipf_xmin * ((1.0 - u) ** (-1.0 / max(1e-6, self.zipf_s - 1.0))))
        return min(max(self.zipf_xmin, x), self.zipf_max)

    def _sample_class_id(self) -> int:
        return self.rng.randrange(self.service_classes)

    def _build_request(self) -> Request:
        request = Request(
            rid=self._rid,
            t_arrival=self.env.now,
            class_id=self._sample_class_id(),
            hidden_size=self._sample_hidden_size(),
        )
        self._rid += 1
        return request

    def _current_gamma_params(self, now: float) -> Tuple[float, float]:
        # Format: [(window_end_time, alpha, beta), ...]
        for window_end, alpha, beta in self.gamma_windows:
            if now < window_end:
                return alpha, beta

        if self.gamma_windows:
            _, alpha, beta = self.gamma_windows[-1]
            return alpha, beta

        return 2.0, 0.7

    def run(self):
        if self.arrival_mode == "trace_replay":
            yield from self._run_trace_replay()
            return
        if self.arrival_mode == "modeled_gamma":
            yield from self._run_modeled_gamma()
            return
        raise ValueError(f"Unsupported arrival_mode: {self.arrival_mode}")

    def _run_trace_replay(self):
        last_t = 0.0
        for timestamp in self.trace_timestamps:
            if timestamp > self.t_end:
                break
            delta = max(0.0, timestamp - last_t)
            if delta > 0:
                yield self.env.timeout(delta)
            last_t = timestamp
            self.on_request(self._build_request())

    def _run_modeled_gamma(self):
        while self.env.now < self.t_end:
            alpha, beta = self._current_gamma_params(self.env.now)
            inter_arrival = self.rng.gammavariate(alpha, beta)
            if inter_arrival <= 0:
                continue

            yield self.env.timeout(inter_arrival)
            if self.env.now > self.t_end:
                break

            self.on_request(self._build_request())


def default_gamma_windows(
    t_end: float, window_size: float = 20 * 60
) -> List[Tuple[float, float, float]]:
    """Create 20-minute windows with a repeating burst/calm pattern."""

    pattern = [
        (2.5, 0.30),  # More bursty: mean inter-arrival ~ 0.75s
        (2.0, 0.65),  # Medium load: mean ~ 1.30s
        (1.8, 1.10),  # Lower load: mean ~ 1.98s
    ]

    windows: List[Tuple[float, float, float]] = []
    current_end = 0.0
    pattern_idx = 0

    while current_end < t_end:
        current_end += window_size
        alpha, beta = pattern[pattern_idx % len(pattern)]
        windows.append((current_end, alpha, beta))
        pattern_idx += 1

    return windows


def load_trace_csv(path: Path) -> List[float]:
    """Load request timestamps (seconds) from a CSV-like file."""

    timestamps: List[float] = []
    with path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            try:
                timestamp = float(row[0].strip())
            except ValueError:
                # Skip header or invalid lines.
                continue
            if timestamp >= 0:
                timestamps.append(timestamp)

    return sorted(timestamps)

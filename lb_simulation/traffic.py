"""Traffic generation for bursty and trace-replay arrivals."""

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import simpy

from .models import Request


@dataclass
class ServiceClassTrafficSpec:
    """Traffic specification for one service class."""

    class_id: int
    arrival_mode: str
    zipf_s: float = 1.20
    zipf_xmin: int = 16
    zipf_max: int = 2048
    gamma_windows: List[Tuple[float, float, float]] = field(default_factory=list)
    trace_timestamps: List[float] = field(default_factory=list)


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
        fixed_class_id: Optional[int] = None,
        next_rid: Optional[Callable[[], int]] = None,
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
        self.fixed_class_id = fixed_class_id
        self.next_rid = next_rid
        self._rid = 0

    def _sample_job_size(self) -> int:
        # Inverse transform sampling for a truncated discrete Zipf-like distribution.
        u = self.rng.random()
        x = int(self.zipf_xmin * ((1.0 - u) ** (-1.0 / max(1e-6, self.zipf_s - 1.0))))
        return min(max(self.zipf_xmin, x), self.zipf_max)

    def _sample_class_id(self) -> int:
        if self.fixed_class_id is not None:
            return self.fixed_class_id
        return self.rng.randrange(self.service_classes)

    def _build_request(self) -> Request:
        rid = self.next_rid() if self.next_rid else self._rid
        request = Request(
            rid=rid,
            t_arrival=self.env.now,
            class_id=self._sample_class_id(),
            job_size=self._sample_job_size(),
        )
        if self.next_rid is None:
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


def constant_gamma_windows(
    t_end: float, alpha: float, beta: float, window_size: float = 20 * 60
) -> List[Tuple[float, float, float]]:
    """Create repeated windows with fixed gamma parameters."""

    windows: List[Tuple[float, float, float]] = []
    current_end = 0.0
    while current_end < t_end:
        current_end += window_size
        windows.append((current_end, alpha, beta))
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


def _parse_gamma_windows(raw: Any) -> List[Tuple[float, float, float]]:
    """Parse gamma windows from a list of dicts/tuples."""

    if not isinstance(raw, list):
        raise ValueError("gamma_windows must be a list.")

    windows: List[Tuple[float, float, float]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            try:
                window_end = float(item["window_end"])
                alpha = float(item["alpha"])
                beta = float(item["beta"])
            except KeyError as error:
                raise ValueError(
                    f"gamma_windows[{idx}] is missing key: {error.args[0]}"
                ) from error
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            window_end = float(item[0])
            alpha = float(item[1])
            beta = float(item[2])
        else:
            raise ValueError(
                "Each gamma_windows item must be {'window_end','alpha','beta'} or [end,alpha,beta]."
            )

        windows.append((window_end, alpha, beta))

    windows.sort(key=lambda x: x[0])
    return windows


def _resolve_trace_path(config_dir: Path, trace_value: str) -> Path:
    """Resolve trace file path, supporting both absolute and relative values."""

    trace_path = Path(trace_value)
    if not trace_path.is_absolute():
        trace_path = config_dir / trace_path
    return trace_path


def load_service_class_config(path: Path, t_end: float) -> List[ServiceClassTrafficSpec]:
    """
    Load per-class traffic specs from JSON.

    Supported schema:
    {
      "classes": [
        {
          "class_id": 0,
          "arrival_mode": "trace_replay" | "modeled_gamma",
          "trace_file": "traces/class0.csv",
          "gamma_windows": [{"window_end": 1200, "alpha": 2.5, "beta": 0.3}],
          "gamma": {"alpha": 2.5, "beta": 0.3, "window_size": 1200},
          "zipf": {"s": 1.2, "xmin": 16, "max": 2048}
        }
      ]
    }
    """

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        raw_classes = payload
    elif isinstance(payload, dict) and isinstance(payload.get("classes"), list):
        raw_classes = payload["classes"]
    else:
        raise ValueError("Service class config must be a list or an object with 'classes'.")

    config_dir = path.parent
    specs: List[ServiceClassTrafficSpec] = []
    seen_class_ids = set()

    for idx, item in enumerate(raw_classes):
        if not isinstance(item, dict):
            raise ValueError(f"classes[{idx}] must be an object.")

        class_id = int(item.get("class_id", idx))
        if class_id in seen_class_ids:
            raise ValueError(f"Duplicate class_id found: {class_id}")
        seen_class_ids.add(class_id)

        arrival_mode = str(item.get("arrival_mode", "modeled_gamma")).strip().lower()

        zipf_cfg = item.get("zipf", {})
        if zipf_cfg is None:
            zipf_cfg = {}
        if not isinstance(zipf_cfg, dict):
            raise ValueError(f"classes[{idx}].zipf must be an object.")

        zipf_s = float(zipf_cfg.get("s", 1.20))
        zipf_xmin = int(zipf_cfg.get("xmin", 16))
        zipf_max = int(zipf_cfg.get("max", 2048))

        trace_timestamps: List[float] = []
        gamma_windows: List[Tuple[float, float, float]] = []

        if arrival_mode == "trace_replay":
            trace_file = item.get("trace_file")
            if not trace_file:
                raise ValueError(
                    f"classes[{idx}] requires 'trace_file' when arrival_mode='trace_replay'."
                )
            trace_path = _resolve_trace_path(config_dir, str(trace_file))
            trace_timestamps = load_trace_csv(trace_path)

        elif arrival_mode == "modeled_gamma":
            if item.get("gamma_windows") is not None:
                gamma_windows = _parse_gamma_windows(item["gamma_windows"])
            elif isinstance(item.get("gamma"), dict):
                gamma_cfg = item["gamma"]
                try:
                    alpha = float(gamma_cfg["alpha"])
                    beta = float(gamma_cfg["beta"])
                except KeyError as error:
                    raise ValueError(
                        f"classes[{idx}].gamma is missing key: {error.args[0]}"
                    ) from error
                window_size = float(gamma_cfg.get("window_size", 20 * 60))
                gamma_windows = constant_gamma_windows(
                    t_end=t_end,
                    alpha=alpha,
                    beta=beta,
                    window_size=window_size,
                )
            else:
                gamma_windows = default_gamma_windows(t_end)
        else:
            raise ValueError(
                f"classes[{idx}] has unsupported arrival_mode: {arrival_mode}"
            )

        specs.append(
            ServiceClassTrafficSpec(
                class_id=class_id,
                arrival_mode=arrival_mode,
                zipf_s=zipf_s,
                zipf_xmin=zipf_xmin,
                zipf_max=zipf_max,
                gamma_windows=gamma_windows,
                trace_timestamps=trace_timestamps,
            )
        )

    return specs

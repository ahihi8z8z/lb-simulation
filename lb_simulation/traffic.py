"""Traffic generation for bursty and trace-replay arrivals."""

import csv
import json
import logging
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import simpy

from .models import Request

logger = logging.getLogger(__name__)


@dataclass
class TraceRecord:
    """One request record parsed from trace CSV."""

    timestamp: float
    total_tokens: int
    model: str
    log_type: str


@dataclass
class ServiceClassTrafficSpec:
    """Traffic specification for one service class."""

    class_id: int
    arrival_mode: str
    model: str = "modeled"
    log_type: str = "modeled_gamma"
    trace_traffic_scale: int = 1
    zipf_s: float = 1.20
    zipf_xmin: int = 16
    zipf_max: int = 2048
    response_slope: float = 1.0
    response_intercept: float = 0.0
    gamma_windows: List[Tuple[float, float, float]] = field(default_factory=list)
    trace_records: List[TraceRecord] = field(default_factory=list)


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
        trace_traffic_scale: int = 1,
        zipf_s: float = 1.20,
        zipf_xmin: int = 16,
        zipf_max: int = 2048,
        response_slope: float = 1.0,
        response_intercept: float = 0.0,
        gamma_windows: Optional[List[Tuple[float, float, float]]] = None,
        trace_records: Optional[List[TraceRecord]] = None,
        model: str = "modeled",
        log_type: str = "modeled_gamma",
        fixed_class_id: Optional[int] = None,
        next_rid: Optional[Callable[[], int]] = None,
    ) -> None:
        self.env = env
        self.t_end = t_end
        self.arrival_mode = arrival_mode
        self.on_request = on_request
        self.rng = rng or random.Random()
        self.service_classes = max(1, service_classes)
        self.trace_traffic_scale = max(1, int(trace_traffic_scale))

        self.zipf_s = zipf_s
        self.zipf_xmin = zipf_xmin
        self.zipf_max = zipf_max
        self.response_slope = response_slope
        self.response_intercept = response_intercept

        self.gamma_windows = gamma_windows or []
        self.trace_records = trace_records or []
        self.model = model
        self.log_type = log_type
        self.fixed_class_id = fixed_class_id
        self.next_rid = next_rid
        self._rid = 0
        logger.info(
            "TrafficGenerator initialized mode=%s t_end=%.3f class_id=%s trace_scale=%d",
            self.arrival_mode,
            self.t_end,
            self.fixed_class_id,
            self.trace_traffic_scale,
        )

    def _sample_request_length(self) -> int:
        # Inverse transform sampling for a truncated discrete Zipf-like distribution.
        u = self.rng.random()
        x = int(self.zipf_xmin * ((1.0 - u) ** (-1.0 / max(1e-6, self.zipf_s - 1.0))))
        return min(max(self.zipf_xmin, x), self.zipf_max)

    def _sample_modeled_gamma_job_size(self) -> int:
        request_length = self._sample_request_length()
        response_length = int(round(self.response_slope * request_length + self.response_intercept))
        response_length = max(0, response_length)
        return request_length + response_length

    def _sample_class_id(self) -> int:
        if self.fixed_class_id is not None:
            return self.fixed_class_id
        return self.rng.randrange(self.service_classes)

    def _build_request(
        self,
        job_size: Optional[int] = None,
        model: Optional[str] = None,
        log_type: Optional[str] = None,
    ) -> Request:
        rid = self.next_rid() if self.next_rid else self._rid
        request = Request(
            rid=rid,
            t_arrival=self.env.now,
            class_id=self._sample_class_id(),
            job_size=(
                job_size
                if job_size is not None
                else self._sample_modeled_gamma_job_size()
            ),
            model=model if model is not None else self.model,
            log_type=log_type if log_type is not None else self.log_type,
        )
        if self.next_rid is None:
            self._rid += 1
        logger.debug(
            "Built request rid=%d class=%d job_size=%d model=%s log_type=%s",
            request.rid,
            request.class_id,
            request.job_size,
            request.model,
            request.log_type,
        )
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
        for record in self.trace_records:
            if record.timestamp > self.t_end:
                break
            delta = max(0.0, record.timestamp - last_t)
            if delta > 0:
                yield self.env.timeout(delta)
            last_t = record.timestamp
            for _ in range(self.trace_traffic_scale):
                self.on_request(
                    self._build_request(
                        job_size=record.total_tokens,
                        model=record.model,
                        log_type=record.log_type,
                    )
                )

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


def _normalize_column_name(name: str) -> str:
    """Normalize header name for robust CSV key lookup."""

    return " ".join(name.strip().lower().replace("_", " ").split())


def _as_int_token_count(raw: str) -> int:
    """Parse token count from integer-like CSV text."""

    value = raw.strip().replace(",", "")
    return int(float(value))


def _canonicalize_trace_model(raw: str) -> str:
    """Validate and canonicalize model values coming from trace."""

    normalized = "".join(raw.strip().lower().split())
    if normalized in {"chatgpt", "chatgpt(gpt-3.5)"}:
        return "ChatGPT"
    if normalized == "gpt-4":
        return "GPT-4"
    raise ValueError(
        "Invalid trace model value. Allowed values are ChatGPT (GPT-3.5) and GPT-4."
    )


def _canonicalize_trace_log_type(raw: str) -> str:
    """Validate and canonicalize log type values coming from trace."""

    normalized = " ".join(raw.strip().lower().split())
    if normalized == "conversation log":
        return "Conversation log"
    if normalized == "api log":
        return "API log"
    raise ValueError(
        "Invalid trace log type value. Allowed values are Conversation log and API log."
    )


def load_trace_csv(path: Path) -> List[TraceRecord]:
    """Load trace records and enforce allowed model/log-type categories."""

    records: List[TraceRecord] = []
    logger.info("Loading trace CSV from %s", path)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Trace file has no header: {path}")

        normalized_to_raw = {
            _normalize_column_name(field): field for field in reader.fieldnames if field
        }

        required = ["timestamp", "total tokens", "model", "log type"]
        missing = [key for key in required if key not in normalized_to_raw]
        if missing:
            raise ValueError(
                f"Trace file is missing required columns {missing}: {path}"
            )

        ts_col = normalized_to_raw["timestamp"]
        total_col = normalized_to_raw["total tokens"]
        model_col = normalized_to_raw["model"]
        log_type_col = normalized_to_raw["log type"]

        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            ts_raw = (row.get(ts_col) or "").strip()
            total_raw = (row.get(total_col) or "").strip()
            if not ts_raw or not total_raw:
                continue

            try:
                timestamp = float(ts_raw)
                total_tokens = _as_int_token_count(total_raw)
            except ValueError as error:
                raise ValueError(
                    f"Invalid trace value at row {row_idx} in {path}: {error}"
                ) from error

            if timestamp < 0:
                continue

            model_raw = (row.get(model_col) or "").strip()
            log_type_raw = (row.get(log_type_col) or "").strip()
            if not model_raw or not log_type_raw:
                raise ValueError(
                    f"Trace row {row_idx} in {path} has empty Model or Log Type."
                )

            try:
                model = _canonicalize_trace_model(model_raw)
                log_type = _canonicalize_trace_log_type(log_type_raw)
            except ValueError as error:
                raise ValueError(f"Trace row {row_idx} in {path}: {error}") from error

            records.append(
                TraceRecord(
                    timestamp=timestamp,
                    total_tokens=max(1, total_tokens),
                    model=model,
                    log_type=log_type,
                )
            )

    records.sort(key=lambda rec: rec.timestamp)
    logger.info("Loaded %d trace records from %s", len(records), path)
    return records


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
          "model": "GPT-4",
          "log_type": "Conversation log",
          "trace_file": "traces/class0.csv",
          "traffic_scale": 3,
          "gamma_windows": [{"window_end": 1200, "alpha": 2.5, "beta": 0.3}],
          "gamma": {"alpha": 2.5, "beta": 0.3, "window_size": 1200},
          "zipf": {"s": 1.2, "xmin": 16, "max": 2048},
          "response_linear": {"slope": 0.7, "intercept": 32.0}
        }
      ]
    }
    """

    logger.info("Loading service class config from %s", path)
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
        model = str(item.get("model", "modeled")).strip() or "modeled"
        log_type = str(item.get("log_type", "modeled_gamma")).strip() or "modeled_gamma"
        trace_traffic_scale = 1

        zipf_s = 1.20
        zipf_xmin = 16
        zipf_max = 2048
        response_slope = 1.0
        response_intercept = 0.0

        trace_records: List[TraceRecord] = []
        gamma_windows: List[Tuple[float, float, float]] = []

        if arrival_mode == "trace_replay":
            trace_traffic_scale = int(item.get("traffic_scale", 1))
            if trace_traffic_scale <= 0:
                raise ValueError(
                    f"classes[{idx}].traffic_scale must be a positive integer."
                )
            if item.get("zipf") is not None:
                raise ValueError(
                    f"classes[{idx}] must not set 'zipf' when arrival_mode='trace_replay'."
                )
            if item.get("response_linear") is not None:
                raise ValueError(
                    f"classes[{idx}] must not set 'response_linear' when arrival_mode='trace_replay'."
                )
            trace_file = item.get("trace_file")
            if not trace_file:
                raise ValueError(
                    f"classes[{idx}] requires 'trace_file' when arrival_mode='trace_replay'."
                )
            trace_path = _resolve_trace_path(config_dir, str(trace_file))
            trace_records = load_trace_csv(trace_path)
            total_trace_records = len(trace_records)
            try:
                model = _canonicalize_trace_model(model)
                log_type = _canonicalize_trace_log_type(log_type)
            except ValueError as error:
                raise ValueError(
                    f"classes[{idx}] has invalid model/log_type for trace_replay: {error}"
                ) from error
            trace_records = [
                record
                for record in trace_records
                if record.model == model and record.log_type == log_type
            ]
            if not trace_records:
                warnings.warn(
                    (
                        f"classes[{idx}] (class_id={class_id}) matched 0 records from "
                        f"{trace_path} with model='{model}' and log_type='{log_type}' "
                        f"(total trace rows: {total_trace_records})."
                    ),
                    UserWarning,
                    stacklevel=2,
                )
            elif not any(record.timestamp <= t_end for record in trace_records):
                warnings.warn(
                    (
                        f"classes[{idx}] (class_id={class_id}) matched "
                        f"{len(trace_records)} records from {trace_path}, but none are "
                        f"within t_end={t_end}. First matched timestamp is "
                        f"{trace_records[0].timestamp}."
                    ),
                    UserWarning,
                    stacklevel=2,
                )

        elif arrival_mode == "modeled_gamma":
            if item.get("traffic_scale") is not None:
                raise ValueError(
                    f"classes[{idx}] must not set 'traffic_scale' when arrival_mode='modeled_gamma'."
                )
            zipf_cfg = item.get("zipf", {})
            if zipf_cfg is None:
                zipf_cfg = {}
            if not isinstance(zipf_cfg, dict):
                raise ValueError(f"classes[{idx}].zipf must be an object.")

            zipf_s = float(zipf_cfg.get("s", 1.20))
            zipf_xmin = int(zipf_cfg.get("xmin", 16))
            zipf_max = int(zipf_cfg.get("max", 2048))
            if zipf_xmin <= 0:
                raise ValueError(f"classes[{idx}].zipf.xmin must be > 0.")
            if zipf_max < zipf_xmin:
                raise ValueError(f"classes[{idx}].zipf.max must be >= zipf.xmin.")

            response_cfg = item.get("response_linear", {})
            if response_cfg is None:
                response_cfg = {}
            if not isinstance(response_cfg, dict):
                raise ValueError(f"classes[{idx}].response_linear must be an object.")
            response_slope = float(response_cfg.get("slope", 1.0))
            response_intercept = float(response_cfg.get("intercept", 0.0))
            if response_slope < 0:
                raise ValueError(f"classes[{idx}].response_linear.slope must be >= 0.")

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
                model=model,
                log_type=log_type,
                trace_traffic_scale=trace_traffic_scale,
                zipf_s=zipf_s,
                zipf_xmin=zipf_xmin,
                zipf_max=zipf_max,
                response_slope=response_slope,
                response_intercept=response_intercept,
                gamma_windows=gamma_windows,
                trace_records=trace_records,
            )
        )
        logger.debug(
            "Loaded service class class_id=%d mode=%s trace_records=%d",
            class_id,
            arrival_mode,
            len(trace_records),
        )

    logger.info("Service class config loaded classes=%d", len(specs))
    return specs

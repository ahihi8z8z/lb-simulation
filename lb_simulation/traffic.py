"""Traffic generation for bursty and trace-replay arrivals."""

import json
import logging
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import simpy
import pandas as pd
import numpy as np

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
    description: str = ""
    model: str = "modeled"
    log_type: str = "modeled_gamma"
    trace_traffic_scale: int = 1
    zipf_s: float = 1.20
    zipf_xmin: int = 16
    zipf_max: int = 2048
    response_slope: float = 1.0
    response_intercept: float = 0.0
    seed: Optional[int] = None
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
        self.np_rng = np.random.default_rng(self.rng.randrange(1, 2**31))
        self.service_classes = max(1, service_classes)
        self.trace_traffic_scale = max(1, int(trace_traffic_scale))
        # Tiny spacing used to avoid exact same timestamp for trace-scale replicas.
        self.trace_scale_replica_offset = 1e-3

        self.zipf_s = zipf_s
        self.zipf_param = max(1.000001, float(zipf_s))
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
        if self.arrival_mode == "trace_replay":
            logger.info(
                "TrafficGenerator initialized mode=%s t_end=%.3f class_id=%s trace_scale=%d",
                self.arrival_mode,
                self.t_end,
                self.fixed_class_id,
                self.trace_traffic_scale,
            )
        else:
            logger.info(
                "TrafficGenerator initialized mode=%s t_end=%.3f class_id=%s",
                self.arrival_mode,
                self.t_end,
                self.fixed_class_id,
            )

    def _sample_request_length(self) -> int:
        if self.zipf_max <= self.zipf_xmin:
            return self.zipf_xmin

        sampled = int(self.np_rng.zipf(a=self.zipf_param))
        x = self.zipf_xmin + sampled - 1
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
        if not self.trace_records:
            return
        base_t = self.trace_records[0].timestamp
        for record in self.trace_records:
            relative_t = max(0.0, record.timestamp - base_t)
            if relative_t > self.t_end:
                break
            delta = relative_t - self.env.now
            if delta > 0:
                yield self.env.timeout(delta)
            for replica_idx in range(self.trace_traffic_scale):
                if replica_idx > 0:
                    yield self.env.timeout(self.trace_scale_replica_offset)
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


def _normalize_column_name(name: str) -> str:
    """Normalize header name for robust CSV key lookup."""

    return " ".join(name.strip().lower().replace("_", " ").split())


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

    logger.info("Loading trace CSV from %s", path)
    header_df = pd.read_csv(path, nrows=0)
    fieldnames = [str(field) for field in header_df.columns]
    if not fieldnames:
        raise ValueError(f"Trace file has no header: {path}")

    normalized_to_raw = {
        _normalize_column_name(field): field for field in fieldnames if field
    }
    required = ["timestamp", "total tokens", "model", "log type"]
    missing = [key for key in required if key not in normalized_to_raw]
    if missing:
        raise ValueError(f"Trace file is missing required columns {missing}: {path}")

    ts_col = normalized_to_raw["timestamp"]
    total_col = normalized_to_raw["total tokens"]
    model_col = normalized_to_raw["model"]
    log_type_col = normalized_to_raw["log type"]

    trace_df = pd.read_csv(
        path,
        usecols=[ts_col, total_col, model_col, log_type_col],
        dtype=str,
        keep_default_na=False,
    )
    if trace_df.empty:
        logger.info("Loaded 0 trace records from %s", path)
        return []

    timestamp_raw = trace_df[ts_col].astype(str).str.strip()
    total_raw = trace_df[total_col].astype(str).str.strip()
    non_empty_mask = (timestamp_raw != "") & (total_raw != "")
    filtered_df = trace_df.loc[non_empty_mask].copy()
    if filtered_df.empty:
        logger.info("Loaded 0 trace records from %s", path)
        return []

    timestamp_numeric = pd.to_numeric(
        filtered_df[ts_col].astype(str).str.strip(),
        errors="coerce",
    )
    total_numeric = pd.to_numeric(
        filtered_df[total_col].astype(str).str.strip().str.replace(",", "", regex=False),
        errors="coerce",
    )

    invalid_numeric_mask = timestamp_numeric.isna() | total_numeric.isna()
    if invalid_numeric_mask.any():
        first_idx = int(invalid_numeric_mask[invalid_numeric_mask].index[0])
        row_idx = first_idx + 2
        raise ValueError(
            f"Invalid trace value at row {row_idx} in {path}: cannot parse numeric fields."
        )

    valid_timestamp_mask = timestamp_numeric >= 0
    if not valid_timestamp_mask.any():
        logger.info("Loaded 0 trace records from %s", path)
        return []
    filtered_df = filtered_df.loc[valid_timestamp_mask].copy()
    timestamp_numeric = timestamp_numeric.loc[valid_timestamp_mask]
    total_numeric = total_numeric.loc[valid_timestamp_mask]

    model_raw = filtered_df[model_col].astype(str).str.strip()
    log_type_raw = filtered_df[log_type_col].astype(str).str.strip()
    empty_model_or_log_type = (model_raw == "") | (log_type_raw == "")
    if empty_model_or_log_type.any():
        first_idx = int(empty_model_or_log_type[empty_model_or_log_type].index[0])
        row_idx = first_idx + 2
        raise ValueError(f"Trace row {row_idx} in {path} has empty Model or Log Type.")

    model_normalized = model_raw.str.lower().str.replace(r"\s+", "", regex=True)
    model_series = model_normalized.map(
        {
            "chatgpt": "ChatGPT",
            "chatgpt(gpt-3.5)": "ChatGPT",
            "gpt-4": "GPT-4",
        }
    )
    invalid_model_mask = model_series.isna()
    if invalid_model_mask.any():
        first_idx = int(invalid_model_mask[invalid_model_mask].index[0])
        row_idx = first_idx + 2
        raw_model = model_raw.loc[first_idx]
        raise ValueError(
            (
                f"Trace row {row_idx} in {path}: Invalid trace model value '{raw_model}'. "
                "Allowed values are ChatGPT (GPT-3.5) and GPT-4."
            )
        )

    log_type_normalized = log_type_raw.str.lower().str.replace(r"\s+", " ", regex=True)
    log_type_series = log_type_normalized.map(
        {
            "conversation log": "Conversation log",
            "api log": "API log",
        }
    )
    invalid_log_type_mask = log_type_series.isna()
    if invalid_log_type_mask.any():
        first_idx = int(invalid_log_type_mask[invalid_log_type_mask].index[0])
        row_idx = first_idx + 2
        raw_log_type = log_type_raw.loc[first_idx]
        raise ValueError(
            (
                f"Trace row {row_idx} in {path}: Invalid trace log type value '{raw_log_type}'. "
                "Allowed values are Conversation log and API log."
            )
        )

    total_tokens_int = total_numeric.astype(float).astype(int)
    records = [
        TraceRecord(
            timestamp=float(timestamp_value),
            total_tokens=max(1, int(total_tokens_value)),
            model=str(model_value),
            log_type=str(log_type_value),
        )
        for timestamp_value, total_tokens_value, model_value, log_type_value in zip(
            timestamp_numeric.tolist(),
            total_tokens_int.tolist(),
            model_series.tolist(),
            log_type_series.tolist(),
        )
    ]

    records.sort(key=lambda rec: rec.timestamp)
    logger.info("Loaded %d trace records from %s", len(records), path)
    return records


def _resolve_config_path(config_dir: Path, value: str) -> Path:
    """Resolve a config-referenced path, supporting absolute and relative values."""

    path = Path(value)
    if not path.is_absolute():
        path = config_dir / path
    return path


def _parse_positive_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a number.") from error
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return parsed


def _load_gamma_windows_from_file(
    path: Path,
    *,
    scale_gamma: float,
    scale_beta: float,
) -> List[Tuple[float, float, float]]:
    """Load gamma windows from CSV file and apply scaling factors."""

    header_df = pd.read_csv(path, nrows=0)
    fieldnames = [str(field) for field in header_df.columns]
    if not fieldnames:
        raise ValueError(f"Gamma params file has no header: {path}")

    normalized_to_raw = {
        _normalize_column_name(field): field for field in fieldnames if field
    }

    alpha_col = normalized_to_raw.get("alpha") or normalized_to_raw.get("gamma")
    beta_col = normalized_to_raw.get("beta")
    if alpha_col is None or beta_col is None:
        raise ValueError(
            (
                f"Gamma params file {path} must contain columns for alpha/gamma and beta. "
                "Supported names include: alpha, gamma, beta."
            )
        )

    source_mode = ""
    usecols = [alpha_col, beta_col]
    window_end_col = normalized_to_raw.get("window end")
    window_index_col = normalized_to_raw.get("window index")
    window_seconds_col = normalized_to_raw.get("window seconds")
    window_end_ts_col = normalized_to_raw.get("window end timestamp")
    window_start_ts_col = normalized_to_raw.get("window start timestamp")

    if window_end_col is not None:
        source_mode = "window_end"
        usecols.append(window_end_col)
    elif (window_index_col is not None) and (window_seconds_col is not None):
        source_mode = "window_index_and_size"
        usecols.extend([window_index_col, window_seconds_col])
    elif (window_end_ts_col is not None) and (window_start_ts_col is not None):
        source_mode = "timestamp_range"
        usecols.extend([window_end_ts_col, window_start_ts_col])
    else:
        raise ValueError(
            (
                f"Gamma params file {path} must provide one window definition mode: "
                "'window_end', or ('window_index' + 'window_seconds'), or "
                "('window_start_timestamp' + 'window_end_timestamp')."
            )
        )

    gamma_df = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False)
    if gamma_df.empty:
        raise ValueError(f"Gamma params file has no rows: {path}")

    alpha_series = pd.to_numeric(
        gamma_df[alpha_col].astype(str).str.strip(),
        errors="coerce",
    )
    beta_series = pd.to_numeric(
        gamma_df[beta_col].astype(str).str.strip(),
        errors="coerce",
    )

    if source_mode == "window_end":
        window_end_series = pd.to_numeric(
            gamma_df[window_end_col].astype(str).str.strip(),
            errors="coerce",
        )
    elif source_mode == "window_index_and_size":
        window_index_series = pd.to_numeric(
            gamma_df[window_index_col].astype(str).str.strip(),
            errors="coerce",
        )
        window_seconds_series = pd.to_numeric(
            gamma_df[window_seconds_col].astype(str).str.strip(),
            errors="coerce",
        )
        window_end_series = (window_index_series + 1.0) * window_seconds_series
    else:
        window_start_series = pd.to_numeric(
            gamma_df[window_start_ts_col].astype(str).str.strip(),
            errors="coerce",
        )
        window_end_ts_series = pd.to_numeric(
            gamma_df[window_end_ts_col].astype(str).str.strip(),
            errors="coerce",
        )
        valid_start = window_start_series.notna()
        if not valid_start.any():
            raise ValueError(
                f"Gamma params file {path} has no valid window_start_timestamp."
            )
        base_start = float(window_start_series[valid_start].iloc[0])
        window_end_series = window_end_ts_series - base_start

    valid_mask = (
        alpha_series.notna()
        & beta_series.notna()
        & window_end_series.notna()
        & (alpha_series > 0)
        & (beta_series > 0)
        & (window_end_series > 0)
    )
    filtered = pd.DataFrame(
        {
            "window_end": window_end_series[valid_mask].astype(float),
            "alpha": alpha_series[valid_mask].astype(float) * scale_gamma,
            "beta": beta_series[valid_mask].astype(float) * scale_beta,
        }
    )
    filtered = filtered[(filtered["alpha"] > 0) & (filtered["beta"] > 0)]
    if filtered.empty:
        raise ValueError(
            (
                f"Gamma params file {path} contains no valid rows after "
                f"applying scale_gamma={scale_gamma} scale_beta={scale_beta}."
            )
        )

    filtered = filtered.sort_values("window_end")
    windows: List[Tuple[float, float, float]] = []
    for row in filtered.itertuples(index=False):
        windows.append((float(row.window_end), float(row.alpha), float(row.beta)))
    return windows


def _load_zipf_params_from_file(path: Path) -> Tuple[float, int, int]:
    """Load zipf parameters from text file (key=value format)."""

    if not path.exists():
        raise ValueError(f"Zipf params file not found: {path}")

    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key_norm = _normalize_column_name(key)
            values[key_norm] = raw_value.strip()

    if "s" not in values:
        raise ValueError(f"Zipf params file {path} is missing key 's'.")
    if "xmin" not in values:
        raise ValueError(f"Zipf params file {path} is missing key 'xmin'.")
    if "max" not in values:
        raise ValueError(f"Zipf params file {path} is missing key 'max'.")

    try:
        zipf_s = float(values["s"])
    except ValueError as error:
        raise ValueError(f"Zipf params file {path}: invalid s={values['s']!r}.") from error
    try:
        zipf_xmin = int(float(values["xmin"]))
    except ValueError as error:
        raise ValueError(
            f"Zipf params file {path}: invalid xmin={values['xmin']!r}."
        ) from error
    try:
        zipf_max = int(float(values["max"]))
    except ValueError as error:
        raise ValueError(f"Zipf params file {path}: invalid max={values['max']!r}.") from error

    if zipf_s <= 1.0:
        raise ValueError(f"Zipf params file {path}: s must be > 1.")
    if zipf_xmin <= 0:
        raise ValueError(f"Zipf params file {path}: xmin must be > 0.")
    if zipf_max < zipf_xmin:
        raise ValueError(f"Zipf params file {path}: max must be >= xmin.")
    return zipf_s, zipf_xmin, zipf_max


def load_service_class_config(path: Path, t_end: float) -> List[ServiceClassTrafficSpec]:
    """
    Load per-class traffic specs from JSON.

    Supported schema:
    {
      "classes": [
        {
          "class_id": 0,
          "arrival_mode": "trace_replay" | "modeled_gamma",
          "description": "Describe the workload modeled by this traffic class",
          "model": "GPT-4",
          "log_type": "Conversation log",
          "trace_file": "traces/class0.csv",
          "traffic_scale": 3,
          "gamma_params_file": "configs/class1_gamma_windows.csv",
          "zipf_params_file": "configs/class1_zipf_params.txt",
          "scale_gamma": 1.0,
          "scale_beta": 1.0,
          "seed": 12345,
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

        if "worker_ids" in item:
            raise ValueError(
                (
                    f"classes[{idx}] uses deprecated field 'worker_ids'. "
                    "Move worker allow-list to topology.service_class_worker_ids."
                )
            )

        class_id = int(item.get("class_id", idx))
        if class_id in seen_class_ids:
            raise ValueError(f"Duplicate class_id found: {class_id}")
        seen_class_ids.add(class_id)

        arrival_mode = str(item.get("arrival_mode", "modeled_gamma")).strip().lower()
        description = str(item.get("description", "")).strip()
        model = str(item.get("model", "modeled")).strip() or "modeled"
        log_type = str(item.get("log_type", "modeled_gamma")).strip() or "modeled_gamma"
        seed: Optional[int] = None
        seed_raw = item.get("seed")
        if seed_raw is not None:
            if isinstance(seed_raw, bool):
                raise ValueError(f"classes[{idx}].seed must be an integer.")
            try:
                seed = int(seed_raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"classes[{idx}].seed must be an integer.") from error
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
            if item.get("gamma_params_file") is not None:
                raise ValueError(
                    (
                        f"classes[{idx}] must not set 'gamma_params_file' when "
                        "arrival_mode='trace_replay'."
                    )
                )
            if item.get("zipf_params_file") is not None:
                raise ValueError(
                    (
                        f"classes[{idx}] must not set 'zipf_params_file' when "
                        "arrival_mode='trace_replay'."
                    )
                )
            if item.get("scale_gamma") is not None:
                raise ValueError(
                    (
                        f"classes[{idx}] must not set 'scale_gamma' when "
                        "arrival_mode='trace_replay'."
                    )
                )
            if item.get("scale_beta") is not None:
                raise ValueError(
                    (
                        f"classes[{idx}] must not set 'scale_beta' when "
                        "arrival_mode='trace_replay'."
                    )
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
            trace_path = _resolve_config_path(config_dir, str(trace_file))
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
            else:
                first_ts = trace_records[0].timestamp
                last_relative_ts = max(0.0, trace_records[-1].timestamp - first_ts)
                logger.info(
                    (
                        "classes[%d] trace_replay uses relative time from first matched "
                        "request: first_timestamp=%.3f relative_span=%.3f t_end=%.3f"
                    ),
                    idx,
                    first_ts,
                    last_relative_ts,
                    t_end,
                )

        elif arrival_mode == "modeled_gamma":
            if item.get("traffic_scale") is not None:
                raise ValueError(
                    f"classes[{idx}] must not set 'traffic_scale' when arrival_mode='modeled_gamma'."
                )
            if item.get("zipf") is not None:
                raise ValueError(
                    (
                        f"classes[{idx}] no longer supports inline 'zipf'. "
                        "Use 'zipf_params_file' instead."
                    )
                )
            if item.get("gamma_windows") is not None:
                raise ValueError(
                    (
                        f"classes[{idx}] no longer supports inline 'gamma_windows'. "
                        "Use 'gamma_params_file' instead."
                    )
                )
            if item.get("gamma") is not None:
                raise ValueError(
                    (
                        f"classes[{idx}] no longer supports inline 'gamma'. "
                        "Use 'gamma_params_file' instead."
                    )
                )

            gamma_params_file = item.get("gamma_params_file")
            if not gamma_params_file:
                raise ValueError(
                    (
                        f"classes[{idx}] requires 'gamma_params_file' when "
                        "arrival_mode='modeled_gamma'."
                    )
                )
            zipf_params_file = item.get("zipf_params_file")
            if not zipf_params_file:
                raise ValueError(
                    (
                        f"classes[{idx}] requires 'zipf_params_file' when "
                        "arrival_mode='modeled_gamma'."
                    )
                )

            scale_gamma = _parse_positive_float(
                item.get("scale_gamma", 1.0),
                f"classes[{idx}].scale_gamma",
            )
            scale_beta = _parse_positive_float(
                item.get("scale_beta", 1.0),
                f"classes[{idx}].scale_beta",
            )

            gamma_params_path = _resolve_config_path(config_dir, str(gamma_params_file))
            if not gamma_params_path.exists():
                raise ValueError(
                    f"classes[{idx}] gamma_params_file not found: {gamma_params_path}"
                )
            gamma_windows = _load_gamma_windows_from_file(
                gamma_params_path,
                scale_gamma=scale_gamma,
                scale_beta=scale_beta,
            )

            zipf_params_path = _resolve_config_path(config_dir, str(zipf_params_file))
            zipf_s, zipf_xmin, zipf_max = _load_zipf_params_from_file(zipf_params_path)

            response_cfg = item.get("response_linear", {})
            if response_cfg is None:
                response_cfg = {}
            if not isinstance(response_cfg, dict):
                raise ValueError(f"classes[{idx}].response_linear must be an object.")
            response_slope = float(response_cfg.get("slope", 1.0))
            response_intercept = float(response_cfg.get("intercept", 0.0))
            if response_slope < 0:
                raise ValueError(f"classes[{idx}].response_linear.slope must be >= 0.")
        else:
            raise ValueError(
                f"classes[{idx}] has unsupported arrival_mode: {arrival_mode}"
            )

        specs.append(
            ServiceClassTrafficSpec(
                class_id=class_id,
                arrival_mode=arrival_mode,
                description=description,
                model=model,
                log_type=log_type,
                trace_traffic_scale=trace_traffic_scale,
                zipf_s=zipf_s,
                zipf_xmin=zipf_xmin,
                zipf_max=zipf_max,
                response_slope=response_slope,
                response_intercept=response_intercept,
                seed=seed,
                gamma_windows=gamma_windows,
                trace_records=trace_records,
            )
        )
        logger.debug(
            "Loaded service class class_id=%d mode=%s description=%s trace_records=%d",
            class_id,
            arrival_mode,
            description,
            len(trace_records),
        )

    logger.info("Service class config loaded classes=%d", len(specs))
    return specs

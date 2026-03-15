"""Simulation assembly, CLI parsing, and summary printing."""

import argparse
import copy
import json
import logging
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import simpy

from .controller import LATENCY_AWARE_POLICIES, LoadBalancerController, load_controller_config
from .inference_pool import InferencePool
from .logging_utils import (
    configure_logging,
    normalize_log_mode,
    set_simulation_time_provider,
)
from .load_balancer import (
    LoadBalancer,
    supported_policies,
)
from .metrics import MetricsCollector
from .models import Request
from .request_csv_logger import RequestCsvLogger
from .traffic import (
    TrafficGenerator,
    load_service_class_config,
)
from .workers import expand_worker_specs, load_worker_class_config

logger = logging.getLogger(__name__)


_DURATION_PATTERN = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[smhdSMHD]?)\s*$"
)

_POLICY_SHORT_ALIASES: Dict[str, str] = {
    "swrr": "static-wrr",
    "staticwrr": "static-wrr",
    "lpwrr": "lp-wrr",
    "spwrr": "sp-wrr",
    "lc": "least_connection",
    "least": "least_connection",
    "p2c": "power_of_two_choices",
    "p2": "power_of_two_choices",
    "pow2": "power_of_two_choices",
    "mema": "min_ema_latency",
    "minema": "min_ema_latency",
    "lp2c": "latency_p2c",
    "latp2c": "latency_p2c",
}
_WRR_POLICIES = frozenset({"static-wrr", "lp-wrr", "sp-wrr"})


def _build_policy_alias_map() -> Dict[str, str]:
    """Build alias -> canonical policy map from registered policies."""

    alias_map: Dict[str, str] = {}
    canonical_names = [name.strip().lower() for name in supported_policies()]
    for canonical in canonical_names:
        alias_map[canonical] = canonical
        alias_map[canonical.replace("_", "-")] = canonical
        alias_map[canonical.replace("_", "")] = canonical
    canonical_set = set(canonical_names)
    for alias, canonical in _POLICY_SHORT_ALIASES.items():
        canonical_key = canonical.strip().lower()
        if canonical_key in canonical_set:
            alias_map[alias.strip().lower()] = canonical_key
    return alias_map


def _policy_alias_pairs() -> List[str]:
    """Return compact alias text entries for CLI help/error messages."""

    alias_map = _build_policy_alias_map()
    pairs: List[str] = []
    for alias in sorted(_POLICY_SHORT_ALIASES):
        canonical = alias_map.get(alias)
        if canonical is not None:
            pairs.append(f"{alias}={canonical}")
    return pairs


def normalize_policy_name(raw: str) -> str:
    """Resolve canonical policy name from canonical value or alias."""

    key = str(raw).strip().lower()
    if not key:
        raise ValueError("Policy must be a non-empty string.")

    alias_map = _build_policy_alias_map()
    canonical = alias_map.get(key)
    if canonical is not None:
        return canonical

    canonical_names = ", ".join(sorted(set(alias_map.values())))
    alias_text = ", ".join(_policy_alias_pairs())
    raise ValueError(
        f"Unknown policy: {raw}. Supported policies: {canonical_names}. "
        f"Aliases: {alias_text}."
    )


def _parse_policy_arg(raw: str) -> str:
    """Argparse adapter for policy alias parsing."""

    try:
        return normalize_policy_name(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_duration_seconds(raw: str) -> float:
    """
    Parse duration text to seconds.

    Supported examples:
    - "300" (seconds)
    - "90s"
    - "1m"
    - "2h"
    - "3d"
    """

    match = _DURATION_PATTERN.match(str(raw))
    if not match:
        raise ValueError(
            f"Invalid --t-end value: {raw}. Supported forms: 300, 90s, 1m, 2h, 3d."
        )

    value = float(match.group("value"))
    unit = match.group("unit").lower()
    multiplier = {
        "": 1.0,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }[unit]
    seconds = value * multiplier
    if seconds <= 0:
        raise ValueError("--t-end must be > 0.")
    return seconds


def _create_run_dir(logs_root: Path, run_prefix: str = "run") -> Path:
    """Create a unique run directory under logs_root."""

    logs_root.mkdir(parents=True, exist_ok=True)
    prefix = str(run_prefix).strip() or "run"
    run_stamp = datetime.now().strftime(f"{prefix}-%Y%m%d-%H%M%S")
    run_dir = logs_root / run_stamp
    suffix = 1
    while run_dir.exists():
        run_dir = logs_root / f"{run_stamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: object, sort_keys: bool = True) -> None:
    """Write a JSON file with stable formatting."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def _load_json_payload(path: Optional[Path]) -> Dict[str, object]:
    """Load one JSON object file; return empty object when missing."""

    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


def _load_unified_config_payload(path: Path) -> Dict[str, object]:
    """Load top-level unified simulation config."""

    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Unified config must be a JSON object.")
    return payload


def _extract_unified_block(
    payload: Dict[str, object],
    key: str,
    required: bool,
    allow_list: bool,
) -> Optional[object]:
    raw = payload.get(key)
    if raw is None:
        if required:
            raise ValueError(f"Missing '{key}' in unified config.")
        return None
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    if allow_list and isinstance(raw, list):
        return copy.deepcopy(raw)
    expected = "object or list" if allow_list else "object"
    raise ValueError(f"'{key}' in unified config must be a JSON {expected}.")


def _resolve_service_trace_paths(service_payload: object, config_dir: Path) -> object:
    """Resolve relative service file paths from unified config service block."""

    normalized = copy.deepcopy(service_payload)
    classes: List[object] = []
    if isinstance(normalized, dict):
        raw_classes = normalized.get("classes")
        if isinstance(raw_classes, list):
            classes = raw_classes
    elif isinstance(normalized, list):
        classes = normalized

    for item in classes:
        if not isinstance(item, dict):
            continue
        for key in ("trace_file", "gamma_params_file", "zipf_params_file"):
            raw_value = item.get(key)
            if not isinstance(raw_value, str):
                continue
            value = raw_value.strip()
            if not value:
                continue
            file_path = Path(value)
            if file_path.is_absolute():
                continue
            item[key] = str((config_dir / file_path).resolve())
    return normalized


def policy_controller_mode(policy: str) -> str:
    """Return controller handling mode for a given LB policy.

    Modes:
    - "none": ignore controller block completely.
    - "latency_only": use only latency_tracker block.
    - "full": use full controller block.
    """

    normalized = normalize_policy_name(policy)
    if normalized in {"lp-wrr", "sp-wrr"}:
        return "full"
    if normalized in LATENCY_AWARE_POLICIES:
        return "latency_only"
    return "none"


def _materialize_unified_config_inputs(
    config_path: Path,
    run_dir: Path,
    controller_mode: str,
) -> Tuple[Path, Path, Optional[Path], Optional[Path], Path]:
    """Materialize service/worker/topology/controller JSON blocks from unified config."""

    payload = _load_unified_config_payload(config_path)
    config_dir = config_path.resolve().parent
    service_payload = _extract_unified_block(
        payload,
        key="service_class",
        required=True,
        allow_list=True,
    )
    worker_payload = _extract_unified_block(
        payload,
        key="worker_class",
        required=True,
        allow_list=True,
    )
    topology_payload = _extract_unified_block(
        payload,
        key="topology",
        required=False,
        allow_list=False,
    )
    controller_payload: Optional[object] = None
    if controller_mode != "none":
        controller_payload = _extract_unified_block(
            payload,
            key="controller",
            required=False,
            allow_list=False,
        )
        if controller_mode == "latency_only":
            if isinstance(controller_payload, dict):
                latency_tracker_payload = controller_payload.get("latency_tracker")
                if latency_tracker_payload is None:
                    controller_payload = {}
                else:
                    controller_payload = {
                        "latency_tracker": copy.deepcopy(latency_tracker_payload)
                    }
            else:
                controller_payload = None
    if service_payload is None:
        raise ValueError("Missing 'service_class' in unified config.")
    if worker_payload is None:
        raise ValueError("Missing 'worker_class' in unified config.")

    resolved_service_payload = _resolve_service_trace_paths(service_payload, config_dir)
    service_path = run_dir / "service_class_config.json"
    worker_path = run_dir / "worker_class_config.json"
    topology_path: Optional[Path] = None
    controller_path: Optional[Path] = None

    _write_json(service_path, resolved_service_payload)
    _write_json(worker_path, worker_payload)
    if topology_payload is not None:
        topology_path = run_dir / "topology_config.json"
        _write_json(topology_path, topology_payload)
    if controller_payload is not None:
        controller_path = run_dir / "controller_config.json"
        _write_json(controller_path, controller_payload)

    unified_snapshot_path = run_dir / "simulation_config.json"
    shutil.copy2(config_path, unified_snapshot_path)
    return service_path, worker_path, topology_path, controller_path, unified_snapshot_path


def _parse_topology_worker_ids(
    topology_payload: Dict[str, object],
    class_ids: List[int],
    num_workers: int,
) -> Dict[int, List[int]]:
    """Parse optional class->worker allow-list topology from unified config."""

    mapping_raw = topology_payload.get("service_class_worker_ids", {})
    if mapping_raw is None:
        mapping_raw = {}
    if not isinstance(mapping_raw, dict):
        raise ValueError("topology.service_class_worker_ids must be an object.")

    known_class_ids = set(class_ids)
    parsed: Dict[int, List[int]] = {}
    for raw_class_id, raw_worker_ids in mapping_raw.items():
        if isinstance(raw_class_id, bool):
            raise ValueError("topology.service_class_worker_ids keys must be class ids.")
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid class id in topology.service_class_worker_ids: {raw_class_id!r}."
            ) from error
        if class_id not in known_class_ids:
            raise ValueError(
                f"topology.service_class_worker_ids has unknown class_id={class_id}."
            )
        if not isinstance(raw_worker_ids, list):
            raise ValueError(
                f"topology.service_class_worker_ids[{class_id}] must be a list."
            )
        if not raw_worker_ids:
            raise ValueError(
                f"topology.service_class_worker_ids[{class_id}] must not be empty."
            )

        worker_ids: List[int] = []
        seen_worker_ids = set()
        for idx, raw_worker_id in enumerate(raw_worker_ids):
            if isinstance(raw_worker_id, bool):
                raise ValueError(
                    f"topology.service_class_worker_ids[{class_id}][{idx}] must be an integer."
                )
            if isinstance(raw_worker_id, int):
                worker_id = raw_worker_id
            elif isinstance(raw_worker_id, str):
                token = raw_worker_id.strip()
                if not token:
                    raise ValueError(
                        f"topology.service_class_worker_ids[{class_id}][{idx}] must be an integer."
                    )
                if token[0] in {"+", "-"}:
                    digits = token[1:]
                else:
                    digits = token
                if not digits.isdigit():
                    raise ValueError(
                        f"topology.service_class_worker_ids[{class_id}][{idx}] must be an integer."
                    )
                worker_id = int(token)
            else:
                raise ValueError(
                    f"topology.service_class_worker_ids[{class_id}][{idx}] must be an integer."
                )

            if worker_id < 0 or worker_id >= num_workers:
                raise ValueError(
                    f"topology.service_class_worker_ids[{class_id}][{idx}]={worker_id} "
                    f"is out of range [0, {num_workers - 1}]."
                )
            if worker_id in seen_worker_ids:
                raise ValueError(
                    f"topology.service_class_worker_ids[{class_id}] contains duplicate worker id "
                    f"{worker_id}."
                )
            seen_worker_ids.add(worker_id)
            worker_ids.append(worker_id)

        parsed[class_id] = worker_ids

    return parsed


def _parse_topology_worker_weights(
    topology_payload: Dict[str, object],
    class_ids: List[int],
    num_workers: int,
    worker_ids_by_class: Dict[int, List[int]],
) -> Dict[int, List[float]]:
    """Parse optional class->worker initial WRR weights from topology config."""

    mapping_raw = topology_payload.get("service_class_worker_weights", {})
    if mapping_raw is None:
        mapping_raw = {}
    if not isinstance(mapping_raw, dict):
        raise ValueError("topology.service_class_worker_weights must be an object.")

    known_class_ids = set(class_ids)
    all_workers = list(range(num_workers))
    parsed: Dict[int, List[float]] = {}
    for raw_class_id, raw_weights in mapping_raw.items():
        if isinstance(raw_class_id, bool):
            raise ValueError("topology.service_class_worker_weights keys must be class ids.")
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid class id in topology.service_class_worker_weights: {raw_class_id!r}."
            ) from error
        if class_id not in known_class_ids:
            raise ValueError(
                f"topology.service_class_worker_weights has unknown class_id={class_id}."
            )
        if not isinstance(raw_weights, dict):
            raise ValueError(
                f"topology.service_class_worker_weights[{class_id}] must be an object "
                "mapping worker_id -> weight."
            )

        connected_worker_ids = worker_ids_by_class.get(class_id, all_workers)
        connected_worker_id_set = set(connected_worker_ids)
        parsed_weights_by_worker: Dict[int, float] = {}
        for raw_worker_id, raw_weight in raw_weights.items():
            if isinstance(raw_worker_id, bool):
                raise ValueError(
                    f"topology.service_class_worker_weights[{class_id}] keys must be worker ids."
                )
            try:
                worker_id = int(raw_worker_id)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    (
                        f"Invalid worker id in topology.service_class_worker_weights[{class_id}]: "
                        f"{raw_worker_id!r}."
                    )
                ) from error
            if worker_id not in connected_worker_id_set:
                raise ValueError(
                    (
                        f"topology.service_class_worker_weights[{class_id}] contains worker_id="
                        f"{worker_id} not in connected workers {sorted(connected_worker_ids)}."
                    )
                )
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    (
                        f"Invalid weight for topology.service_class_worker_weights[{class_id}]"
                        f"[{worker_id}]: {raw_weight!r}."
                    )
                ) from error
            if weight <= 0:
                raise ValueError(
                    (
                        f"topology.service_class_worker_weights[{class_id}][{worker_id}] "
                        "must be > 0."
                    )
                )
            if worker_id in parsed_weights_by_worker:
                raise ValueError(
                    (
                        f"topology.service_class_worker_weights[{class_id}] has duplicate "
                        f"worker_id={worker_id}."
                    )
                )
            parsed_weights_by_worker[worker_id] = weight

        missing_worker_ids = [
            worker_id for worker_id in connected_worker_ids if worker_id not in parsed_weights_by_worker
        ]
        if missing_worker_ids:
            raise ValueError(
                (
                    f"topology.service_class_worker_weights[{class_id}] missing weights for "
                    f"connected workers: {missing_worker_ids}."
                )
            )

        full_weights = [1.0 for _ in range(num_workers)]
        for worker_id, weight in parsed_weights_by_worker.items():
            full_weights[worker_id] = weight
        parsed[class_id] = full_weights

    return parsed


def _build_grouped_summary_payload(
    metrics_summary: Dict[str, object],
    run_config_payload: Dict[str, object],
    service_config_payload: Dict[str, object],
    worker_config_payload: Dict[str, object],
    topology_config_payload: Dict[str, object],
    controller_config_payload: Dict[str, object],
    unified_config_payload: Dict[str, object],
) -> Dict[str, object]:
    """Build grouped summary payload for run logs."""

    return {
        "meta": {
            "policy": metrics_summary.get("policy", ""),
            "arrival_mode": metrics_summary.get("arrival_mode", ""),
            "service_classes": metrics_summary.get("service_classes", 0),
            "worker_classes": metrics_summary.get("worker_classes", 0),
            "workers": metrics_summary.get("workers", 0),
            "lb_workers": metrics_summary.get("lb_workers", 0),
            "detail_enabled": metrics_summary.get("detail_enabled", False),
            "logger_mode": metrics_summary.get("logger_mode", ""),
        },
        "outcomes": {
            "traffic": {
                "dispatched": metrics_summary.get("dispatched", 0),
                "completed": metrics_summary.get("completed", 0),
                "dropped": metrics_summary.get("dropped", 0),
                "drop_rate": metrics_summary.get("drop_rate", 0.0),
                "throughput": metrics_summary.get("throughput", 0.0),
            },
            "latency": {
                "mean": metrics_summary.get("mean_latency", 0.0),
                "median": metrics_summary.get("median_latency", 0.0),
                "p95": metrics_summary.get("p95_latency", 0.0),
                "p99": metrics_summary.get("p99_latency", 0.0),
            },
            "queueing": {
                "avg_queue_len": metrics_summary.get("avg_queue_len", 0.0),
                "avg_global_inflight": metrics_summary.get("avg_global_inflight", 0.0),
                "avg_utilization": metrics_summary.get("avg_utilization", 0.0),
            },
            "dispersion": {
                "worker_latency_mean_stddev": metrics_summary.get(
                    "worker_latency_mean_stddev", 0.0
                ),
                "worker_latency_mean_max_gap": metrics_summary.get(
                    "worker_latency_mean_max_gap", 0.0
                ),
                "service_latency_mean_stddev": metrics_summary.get(
                    "service_latency_mean_stddev", 0.0
                ),
                "service_latency_mean_max_gap": metrics_summary.get(
                    "service_latency_mean_max_gap", 0.0
                ),
                "worker_utilization_stddev": metrics_summary.get(
                    "worker_utilization_stddev", 0.0
                ),
                "worker_utilization_max_gap": metrics_summary.get(
                    "worker_utilization_max_gap", 0.0
                ),
            },
            "job_size": {
                "total": metrics_summary.get("total_job_size", 0),
            },
            "time": {
                "sim_time_total": metrics_summary.get("sim_time_total", 0.0),
                "wall_time_total": metrics_summary.get("wall_time_total", 0.0),
                "drain_time": metrics_summary.get("drain_time", 0.0),
            },
        },
        "breakdown": {
            "job_size_by_class": metrics_summary.get("total_job_size_by_class", {}),
            "latency_by_class": metrics_summary.get("latency_by_class", {}),
            "latency_by_worker": metrics_summary.get("latency_by_worker", {}),
            "drop_by_class": metrics_summary.get("drop_by_class", {}),
            "drop_by_worker": metrics_summary.get("drop_by_worker", {}),
            "utilization_by_worker": metrics_summary.get("utilization_by_worker", []),
        },
        "controller": metrics_summary.get("controller", {}),
        "artifacts": {
            "run_dir": metrics_summary.get("run_dir", ""),
            "summary_file": metrics_summary.get("summary_file", ""),
            "runtime_log_file": metrics_summary.get("runtime_log_file", ""),
            "detail_metrics_file": metrics_summary.get("detail_metrics_file", ""),
            "run_config_file": metrics_summary.get("run_config_file", ""),
            "service_class_config_snapshot_file": metrics_summary.get(
                "service_class_config_snapshot_file", ""
            ),
            "worker_class_config_snapshot_file": metrics_summary.get(
                "worker_class_config_snapshot_file", ""
            ),
            "topology_config_snapshot_file": metrics_summary.get(
                "topology_config_snapshot_file", ""
            ),
            "controller_config_snapshot_file": metrics_summary.get(
                "controller_config_snapshot_file", ""
            ),
            "simulation_config_snapshot_file": metrics_summary.get(
                "simulation_config_snapshot_file", ""
            ),
        },
        "configs": {
            "run": run_config_payload,
            "service_class": service_config_payload,
            "worker_class": worker_config_payload,
            "topology": topology_config_payload,
            "controller": controller_config_payload,
            "simulation": unified_config_payload,
        },
    }


def run_simulation(
    t_end: float = 60 * 60,
    policy: str = "static-wrr",
    simulation_config: Optional[Path] = None,
    seed: int = 42,
    detail: bool = False,
    logger_mode: str = "INFO",
    logs_root: Path = Path("logs"),
    run_prefix: str = "run",
) -> Dict[str, object]:
    """Run one simulation and return aggregate metrics."""
    wall_clock_start = time.perf_counter()
    set_simulation_time_provider(None)

    if simulation_config is None:
        raise ValueError("simulation_config is required.")

    normalized_policy = normalize_policy_name(policy)
    controller_mode = policy_controller_mode(normalized_policy)

    run_dir = _create_run_dir(logs_root, run_prefix=run_prefix)
    runtime_log_path = configure_logging(run_dir, mode=logger_mode)
    normalized_logger_mode = normalize_log_mode(logger_mode)

    runtime_service_config_path: Path
    runtime_worker_config_path: Path
    runtime_topology_config_path: Optional[Path]
    runtime_controller_config_path: Optional[Path]
    simulation_config_snapshot_path: Optional[Path] = None

    (
        runtime_service_config_path,
        runtime_worker_config_path,
        runtime_topology_config_path,
        runtime_controller_config_path,
        simulation_config_snapshot_path,
    ) = _materialize_unified_config_inputs(
        simulation_config,
        run_dir,
        controller_mode=controller_mode,
    )

    if (controller_mode == "latency_only") and (runtime_controller_config_path is None):
        raise ValueError(
            f"Policy '{normalized_policy}' requires controller config with "
            "'latency_tracker' block."
        )
    if (controller_mode == "full") and (runtime_controller_config_path is None):
        required_key = "lp-wrr" if normalized_policy == "lp-wrr" else "sp-wrr"
        raise ValueError(
            f"Policy '{normalized_policy}' requires controller config with "
            f"'latency_tracker' and '{required_key}' blocks."
        )

    logger.info(
        "Starting simulation policy=%s t_end=%.3f service_config=%s worker_config=%s",
        policy,
        t_end,
        runtime_service_config_path,
        runtime_worker_config_path,
    )
    logger.debug("Run directory prepared at %s", run_dir)

    rng = random.Random(seed)
    # Keep NumPy-based sampling deterministic for each run.
    np.random.seed(seed)
    env = simpy.Environment()
    set_simulation_time_provider(lambda: env.now)

    class_specs = load_service_class_config(runtime_service_config_path, t_end=t_end)
    if not class_specs:
        raise ValueError("service_class_config does not contain any class entry.")
    effective_service_classes = len(class_specs)
    service_class_descriptions = {
        str(spec.class_id): spec.description
        for spec in class_specs
        if spec.description
    }
    worker_class_specs = load_worker_class_config(runtime_worker_config_path)
    effective_worker_classes = len(worker_class_specs)
    worker_class_descriptions = {
        str(spec.class_id): spec.description
        for spec in worker_class_specs
        if spec.description
    }
    worker_specs = expand_worker_specs(worker_class_specs)
    num_workers = len(worker_specs)
    controller_cfg = load_controller_config(runtime_controller_config_path)
    controller = LoadBalancerController(
        policy=normalized_policy,
        num_workers=num_workers,
        config=controller_cfg,
        rng=random.Random(rng.randrange(1, 2**31)),
    )
    logger.info(
        "Loaded %d service classes, %d worker classes, %d workers",
        effective_service_classes,
        effective_worker_classes,
        num_workers,
    )

    service_config_snapshot_path = run_dir / "service_class_config.json"
    worker_config_snapshot_path = run_dir / "worker_class_config.json"
    topology_config_snapshot_path: Optional[Path] = None
    controller_config_snapshot_path: Optional[Path] = None

    service_config_snapshot_path = runtime_service_config_path
    worker_config_snapshot_path = runtime_worker_config_path
    topology_config_snapshot_path = runtime_topology_config_path
    controller_config_snapshot_path = runtime_controller_config_path

    run_config_path = run_dir / "run_config.json"
    run_config_payload: Dict[str, object] = {
        "simulation_config": str(simulation_config) if simulation_config else "",
        "simulation_config_snapshot_file": (
            str(simulation_config_snapshot_path) if simulation_config_snapshot_path else ""
        ),
        "controller_config": (
            str(runtime_controller_config_path) if runtime_controller_config_path else ""
        ),
        "controller_config_snapshot_file": (
            str(controller_config_snapshot_path) if controller_config_snapshot_path else ""
        ),
        "topology_config": (
            str(runtime_topology_config_path) if runtime_topology_config_path else ""
        ),
        "topology_config_snapshot_file": (
            str(topology_config_snapshot_path) if topology_config_snapshot_path else ""
        ),
        "detail": detail,
        "logger_mode": normalized_logger_mode,
        "runtime_log_file": str(runtime_log_path),
        "policy": policy,
        "seed": seed,
        "service_class_config": str(runtime_service_config_path),
        "service_class_config_snapshot_file": str(service_config_snapshot_path),
        "service_class_descriptions": service_class_descriptions,
        "t_end": t_end,
        "worker_class_config": str(runtime_worker_config_path),
        "worker_class_config_snapshot_file": str(worker_config_snapshot_path),
        "worker_class_descriptions": worker_class_descriptions,
        "worker_classes": effective_worker_classes,
        "workers": num_workers,
        "lb_workers": num_workers + (1 if controller.latency_tracker_enabled else 0),
    }
    _write_json(run_config_path, run_config_payload)

    service_config_payload = _load_json_payload(service_config_snapshot_path)
    worker_config_payload = _load_json_payload(worker_config_snapshot_path)
    topology_config_payload = _load_json_payload(topology_config_snapshot_path)
    controller_config_payload = _load_json_payload(controller_config_snapshot_path)
    unified_config_payload = _load_json_payload(simulation_config_snapshot_path)

    metrics = MetricsCollector(num_workers=num_workers)
    detail_writer: Optional[RequestCsvLogger] = None
    detail_path: Optional[Path] = None

    if detail:
        detail_path = run_dir / "request_detail_metrics.csv"
        detail_writer = RequestCsvLogger(detail_path)
        detail_writer.open()
        logger.info("Detail metrics enabled at %s", detail_path)

    topology_worker_ids_by_class: Dict[int, List[int]] = {}
    topology_worker_weights_by_class: Dict[int, List[float]] = {}
    if topology_config_payload:
        topology_worker_ids_by_class = _parse_topology_worker_ids(
            topology_payload=topology_config_payload,
            class_ids=[int(spec.class_id) for spec in class_specs],
            num_workers=num_workers,
        )
        topology_worker_weights_by_class = _parse_topology_worker_weights(
            topology_payload=topology_config_payload,
            class_ids=[int(spec.class_id) for spec in class_specs],
            num_workers=num_workers,
            worker_ids_by_class=topology_worker_ids_by_class,
        )

    load_balancers_by_class: Dict[int, LoadBalancer] = {}
    for spec in class_specs:
        class_id = int(spec.class_id)
        if class_id in load_balancers_by_class:
            raise ValueError(f"Duplicate service class id: {class_id}")
        allowed_worker_ids = topology_worker_ids_by_class.get(class_id)
        load_balancers_by_class[class_id] = LoadBalancer(
            num_workers=num_workers,
            policy=normalized_policy,
            worker_ids=allowed_worker_ids,
            lb_id=f"class_{class_id}",
            rng=random.Random(rng.randrange(1, 2**31)),
        )
        if (normalized_policy in _WRR_POLICIES) and (class_id in topology_worker_weights_by_class):
            load_balancers_by_class[class_id].set_worker_weights(
                topology_worker_weights_by_class[class_id]
            )
            logger.info(
                "Applied topology initial WRR weights for class_id=%d",
                class_id,
            )
    controller.initialize(load_balancers_by_class)

    def on_complete(
        request: Request,
        worker_id: int,
        latency: float,
        latency_tracked: bool,
    ) -> None:
        controller.on_request_complete(
            request=request,
            worker_id=worker_id,
            latency=latency,
            latency_tracked=latency_tracked,
        )

    inference_pool = InferencePool(
        env=env,
        worker_specs=worker_specs,
        metrics=metrics,
        on_complete=on_complete,
        on_request_done=detail_writer.write if detail_writer else None,
        rng=rng,
    )

    def _build_detail_state(request: Request) -> Optional[Dict[str, object]]:
        if not detail:
            return None
        class_id = int(request.class_id)
        class_lb = load_balancers_by_class.get(class_id)
        if class_lb is None:
            raise ValueError(f"No load balancer found for class_id={class_id}")
        lb_state = {
            "class_id": class_id,
            "inflight": list(class_lb.inflight),
            "lat_ewma": list(class_lb.lat_ewma),
            "worker_weights": list(class_lb.worker_weights),
            "penalty": list(class_lb.penalty),
            "feedback_count": list(class_lb.feedback_count),
        }
        lb_control_state: Dict[str, object] = {}
        if controller.lb_control_module is not None and controller.lb_control_module.name != "none":
            lb_control_state = controller.lb_control_module.summarize(load_balancers_by_class)
        return {"lb_state": lb_state, "lb_control_state": lb_control_state}

    def on_arrival(request: Request) -> None:
        class_id = int(request.class_id)
        class_lb = load_balancers_by_class.get(class_id)
        if class_lb is None:
            raise ValueError(f"No load balancer found for class_id={class_id}")

        detail_state = _build_detail_state(request)
        lb_selected_worker_id = class_lb.choose_worker(request)
        if controller.is_latency_tracker_worker(lb_selected_worker_id):
            # The tracker worker only forwards to a real worker.
            selected_worker_id = class_lb.consume_redirect_target(request.rid)
            forwarded_worker_id = controller.forward_via_latency_tracker(
                request,
                selected_worker_id=selected_worker_id,
            )
            class_lb.on_dispatch(lb_selected_worker_id)
            class_lb.on_dispatch(forwarded_worker_id)
            inference_pool.dispatch(
                request,
                forwarded_worker_id,
                class_lb,
                latency_tracked=True,
                lb_completion_worker_ids=[
                    forwarded_worker_id,
                    lb_selected_worker_id,
                ],
                lb_selected_worker_id=(
                    selected_worker_id
                    if selected_worker_id is not None
                    else lb_selected_worker_id
                ),
                routed_via_latency_tracker=True,
                detail_state=detail_state,
            )
            logger.debug(
                "Request rid=%d routed via tracker -> worker=%d",
                request.rid,
                forwarded_worker_id,
            )
            return

        class_lb.on_dispatch(lb_selected_worker_id)
        inference_pool.dispatch(
            request,
            lb_selected_worker_id,
            class_lb,
            latency_tracked=False,
            lb_completion_worker_ids=[lb_selected_worker_id],
            lb_selected_worker_id=lb_selected_worker_id,
            routed_via_latency_tracker=False,
            detail_state=detail_state,
        )
        logger.debug(
            "Request rid=%d routed directly -> worker=%d",
            request.rid,
            lb_selected_worker_id,
        )

    rid_counter = 0

    def next_rid() -> int:
        nonlocal rid_counter
        rid = rid_counter
        rid_counter += 1
        return rid

    traffic_generators: List[TrafficGenerator] = []
    for spec in class_specs:
        if (spec.arrival_mode == "modeled_gamma") and (spec.seed is not None):
            class_seed = int(spec.seed)
        else:
            class_seed = rng.randrange(1, 2**31)
        class_rng = random.Random(class_seed)
        logger.debug(
            "Traffic RNG seed class_id=%d mode=%s seed=%d",
            spec.class_id,
            spec.arrival_mode,
            class_seed,
        )
        traffic_generators.append(
            TrafficGenerator(
                env=env,
                t_end=t_end,
                arrival_mode=spec.arrival_mode,
                on_request=on_arrival,
                rng=class_rng,
                service_classes=1,
                trace_traffic_scale=spec.trace_traffic_scale,
                zipf_s=spec.zipf_s,
                zipf_xmin=spec.zipf_xmin,
                zipf_max=spec.zipf_max,
                response_slope=spec.response_slope,
                response_intercept=spec.response_intercept,
                gamma_windows=spec.gamma_windows,
                trace_records=spec.trace_records,
                model=spec.model,
                log_type=spec.log_type,
                fixed_class_id=spec.class_id,
                next_rid=next_rid,
            )
        )

    try:
        for traffic_generator in traffic_generators:
            env.process(traffic_generator.run())
        env.run()
    finally:
        if detail_writer:
            detail_writer.close()

    summary = metrics.summarize(sim_time=env.now, active_time=t_end)
    wall_time_total = time.perf_counter() - wall_clock_start
    sim_time_total = env.now
    drain_time = max(0.0, sim_time_total - t_end)
    summary["wall_time_total"] = wall_time_total
    summary["sim_time_total"] = sim_time_total
    summary["drain_time"] = drain_time
    summary["policy"] = normalized_policy
    summary["workers"] = num_workers
    summary["lb_workers"] = num_workers + (1 if controller.latency_tracker_enabled else 0)
    summary["worker_classes"] = effective_worker_classes
    summary["arrival_mode"] = "per_class_config"
    summary["service_classes"] = effective_service_classes
    summary["simulation_config_file"] = str(simulation_config) if simulation_config else ""
    summary["simulation_config_snapshot_file"] = (
        str(simulation_config_snapshot_path) if simulation_config_snapshot_path else ""
    )
    summary["service_class_config_file"] = str(runtime_service_config_path)
    summary["worker_class_config_file"] = str(runtime_worker_config_path)
    summary["topology_config_file"] = (
        str(runtime_topology_config_path) if runtime_topology_config_path else ""
    )
    summary["controller_config_file"] = (
        str(runtime_controller_config_path) if runtime_controller_config_path else ""
    )
    summary["detail_enabled"] = detail
    summary["detail_metrics_file"] = str(detail_path) if detail_path else ""
    summary["logger_mode"] = normalized_logger_mode
    summary["runtime_log_file"] = str(runtime_log_path)
    summary["run_config_file"] = str(run_config_path)
    summary["run_dir"] = str(run_dir)
    summary["service_class_config_snapshot_file"] = str(service_config_snapshot_path)
    summary["service_class_descriptions"] = service_class_descriptions
    summary["worker_class_config_snapshot_file"] = str(worker_config_snapshot_path)
    summary["worker_class_descriptions"] = worker_class_descriptions
    summary["topology_config_snapshot_file"] = (
        str(topology_config_snapshot_path) if topology_config_snapshot_path else ""
    )
    summary["controller_config_snapshot_file"] = (
        str(controller_config_snapshot_path) if controller_config_snapshot_path else ""
    )
    summary["controller"] = controller.summarize()
    summary_file = run_dir / "summary.json"
    summary["summary_file"] = str(summary_file)
    grouped_summary = _build_grouped_summary_payload(
        metrics_summary=summary,
        run_config_payload=run_config_payload,
        service_config_payload=service_config_payload,
        worker_config_payload=worker_config_payload,
        topology_config_payload=topology_config_payload,
        controller_config_payload=controller_config_payload,
        unified_config_payload=unified_config_payload,
    )
    _write_json(summary_file, grouped_summary, sort_keys=False)
    logger.info(
        (
            "Simulation completed dispatched=%s completed=%s dropped=%s "
            "mean_latency=%.4f wall_time=%.3fs"
        ),
        summary["dispatched"],
        summary["completed"],
        summary["dropped"],
        summary["mean_latency"],
        summary["wall_time_total"],
    )
    set_simulation_time_provider(None)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Event-driven LB simulator (SimPy)")
    policy_choices = ", ".join(supported_policies())
    policy_aliases = ", ".join(_policy_alias_pairs())
    parser.add_argument(
        "-t",
        "--t-end",
        "--time-end",
        dest="t_end",
        type=str,
        default="1h",
        help="Simulation horizon: seconds or duration suffix (e.g. 300, 90s, 1m, 2h, 3d)",
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=_parse_policy_arg,
        metavar="POLICY",
        default="static-wrr",
        help=(
            f"Load balancing policy. Canonical: {policy_choices}. "
            f"Aliases: {policy_aliases}."
        ),
    )
    parser.add_argument(
        "-f",
        "--config",
        "--simulation-config",
        dest="simulation_config",
        type=Path,
        required=True,
        help=(
            "Unified config JSON containing service_class, worker_class, "
            "optional topology, and optional controller."
        ),
    )
    parser.add_argument("-S", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-d",
        "--detail",
        action="store_true",
        help="Write per-request detail metrics to CSV",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        "--logger-mode",
        dest="logger_mode",
        type=str,
        default="INFO",
        help="Logger mode for console/file output (DEBUG or INFO)",
    )
    return parser


def print_summary(summary: Dict[str, object]) -> None:
    """Pretty-print simulation summary."""

    print("=== Simulation Summary ===")
    print(f"policy                : {summary['policy']}")
    print(f"arrival_mode          : {summary['arrival_mode']}")
    print(f"workers               : {summary['workers']}")
    print(f"lb workers            : {summary.get('lb_workers', summary['workers'])}")
    print(f"worker classes        : {summary['worker_classes']}")
    print(f"service classes       : {summary['service_classes']}")
    service_desc = summary.get("service_class_descriptions", {})
    worker_desc = summary.get("worker_class_descriptions", {})
    if isinstance(service_desc, dict) and service_desc:
        print(f"service class desc    : {len(service_desc)} configured")
    if isinstance(worker_desc, dict) and worker_desc:
        print(f"worker class desc     : {len(worker_desc)} configured")
    simulation_config_file = str(summary.get("simulation_config_file", "")).strip()
    print(f"simulation config     : {simulation_config_file}")
    print(f"run dir               : {summary['run_dir']}")
    print(f"run config file       : {summary['run_config_file']}")
    print(f"summary file          : {summary['summary_file']}")
    print(f"dispatched            : {summary['dispatched']}")
    print(f"completed             : {summary['completed']}")
    print(f"dropped               : {summary.get('dropped', 0)}")
    print(f"drop rate             : {summary.get('drop_rate', 0.0):.4%}")
    print(f"total job size        : {summary.get('total_job_size', 0)}")
    print(f"throughput (req/s)    : {summary['throughput']:.4f}")
    print(f"mean latency (s)      : {summary['mean_latency']:.4f}")
    print(f"median latency (s)    : {summary['median_latency']:.4f}")
    print(f"p95 latency (s)       : {summary['p95_latency']:.4f}")
    print(f"p99 latency (s)       : {summary['p99_latency']:.4f}")
    print(f"avg queue len         : {summary['avg_queue_len']:.4f}")
    print(f"avg global inflight   : {summary['avg_global_inflight']:.4f}")
    print(f"avg utilization       : {summary['avg_utilization']:.4f}")
    print(f"sim time total (s)    : {summary['sim_time_total']:.4f}")
    print(f"wall time total (s)   : {summary.get('wall_time_total', 0.0):.4f}")
    print(f"drain time (s)        : {summary['drain_time']:.4f}")
    print(f"detail enabled        : {summary['detail_enabled']}")
    if summary["detail_enabled"]:
        print(f"detail metrics file   : {summary['detail_metrics_file']}")
    print(f"logger mode           : {summary.get('logger_mode', 'INFO')}")
    if summary.get("runtime_log_file"):
        print(f"runtime log file      : {summary['runtime_log_file']}")
    controller = summary.get("controller")
    if isinstance(controller, dict):
        print(f"latency tracker       : {controller.get('latency_tracker_enabled', False)}")
        print(f"latency samples       : {controller.get('latency_samples_total', 0)}")
        print(
            "latency redirects     : "
            f"{controller.get('track_redirected', 0)} / {controller.get('track_decisions', 0)}"
        )
        redirect_policy = controller.get("latency_redirect_policy", {})
        if isinstance(redirect_policy, dict) and redirect_policy.get("name"):
            print(f"redirect policy       : {redirect_policy.get('name')}")
        wrr_control_mode = str(controller.get("wrr_control_mode", "")).strip()
        if not wrr_control_mode:
            policy_name = str(summary.get("policy", "")).strip().lower()
            wrr_control_mode = (
                "fixed"
                if policy_name == "static-wrr"
                else (
                    policy_name
                    if policy_name in {"lp-wrr", "sp-wrr"}
                    else "n/a"
                )
            )
        print(f"wrr control mode      : {wrr_control_mode}")

    by_class = summary.get("latency_by_class", {})
    if isinstance(by_class, dict) and by_class:
        print("\nLatency by class:")
        for class_id, stats in by_class.items():
            print(
                f"  class {class_id}: count={stats['count']} "
                f"mean={stats['mean']:.4f}s p95={stats['p95']:.4f}s"
            )

    total_job_size_by_class = summary.get("total_job_size_by_class", {})
    if isinstance(total_job_size_by_class, dict) and total_job_size_by_class:
        print("\nTotal job size by class:")
        for class_id, total_job_size in total_job_size_by_class.items():
            print(f"  class {class_id}: total_job_size={total_job_size}")

    drop_by_class = summary.get("drop_by_class", {})
    if isinstance(drop_by_class, dict) and drop_by_class:
        print("\nDrop by class:")
        for class_id, stats in drop_by_class.items():
            print(
                f"  class {class_id}: dispatched={stats['dispatched']} "
                f"dropped={stats['dropped']} drop_rate={stats['drop_rate']:.4%}"
            )

    by_worker = summary.get("latency_by_worker", {})
    if isinstance(by_worker, dict) and by_worker:
        print("\nLatency by worker:")
        for worker_id, stats in by_worker.items():
            print(
                f"  worker {worker_id}: count={stats['count']} "
                f"mean={stats['mean']:.4f}s p95={stats['p95']:.4f}s"
            )

    drop_by_worker = summary.get("drop_by_worker", {})
    if isinstance(drop_by_worker, dict) and drop_by_worker:
        print("\nDrop by worker:")
        for worker_id, stats in drop_by_worker.items():
            print(
                f"  worker {worker_id}: dispatched={stats['dispatched']} "
                f"dropped={stats['dropped']} drop_rate={stats['drop_rate']:.4%}"
            )


def main() -> None:
    """CLI main function."""

    args = build_arg_parser().parse_args()
    try:
        t_end_seconds = parse_duration_seconds(args.t_end)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        normalized_mode = normalize_log_mode(args.logger_mode)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    summary = run_simulation(
        t_end=t_end_seconds,
        policy=args.policy,
        simulation_config=args.simulation_config,
        seed=args.seed,
        detail=args.detail,
        logger_mode=normalized_mode,
    )
    print_summary(summary)

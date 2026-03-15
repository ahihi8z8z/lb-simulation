#!/usr/bin/env python3
"""Sweep simulation configs using base_config + scenario overrides + per-scenario sweeps."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lb_simulation.logging_utils import normalize_log_mode
from lb_simulation.runner import (
    normalize_policy_name,
    parse_duration_seconds,
    policy_controller_mode,
    run_simulation,
)

PathSegment = Union[str, int]


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    override: Dict[str, object]
    sweep_spec: Dict[str, object]


@dataclass(frozen=True)
class SweepCase:
    case_id: int
    scenario_name: str
    scenario_case_id: int
    simulation_payload: Dict[str, object]
    selected_values: Dict[str, object]


_MAX_RANGE_POINTS = 10000
_RANGE_KEYS = {"min", "max", "step"}


def _load_json_object(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Config must be a JSON object.")
    return payload


def _resolve_service_trace_paths(payload: Dict[str, object], config_dir: Path) -> Dict[str, object]:
    """Resolve relative service file paths to absolute paths."""

    normalized = copy.deepcopy(payload)
    service_block = normalized.get("service_class")
    classes: List[object] = []
    if isinstance(service_block, dict):
        raw_classes = service_block.get("classes")
        if isinstance(raw_classes, list):
            classes = raw_classes
    elif isinstance(service_block, list):
        classes = service_block

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
            if not file_path.is_absolute():
                item[key] = str((config_dir / file_path).resolve())
    return normalized


def _validate_service_topology_schema(payload: Dict[str, object]) -> None:
    """Reject deprecated per-service worker_ids field."""

    service_block = payload.get("service_class")
    classes: List[object] = []
    if isinstance(service_block, dict):
        raw_classes = service_block.get("classes")
        if isinstance(raw_classes, list):
            classes = raw_classes
    elif isinstance(service_block, list):
        classes = service_block

    for idx, item in enumerate(classes):
        if not isinstance(item, dict):
            continue
        if "worker_ids" in item:
            raise ValueError(
                (
                    f"service_class.classes[{idx}].worker_ids is deprecated. "
                    "Move worker allow-list to topology.service_class_worker_ids."
                )
            )


def _merge_dicts_deep(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    """Deep merge dictionaries; non-dict values replace base values."""

    result: Dict[str, object] = copy.deepcopy(base)
    for key, override_value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = _merge_dicts_deep(base_value, override_value)
        else:
            result[key] = copy.deepcopy(override_value)
    return result


def _path_key(path: Tuple[PathSegment, ...]) -> str:
    if not path:
        return "config"
    parts: List[str] = []
    for segment in path:
        if isinstance(segment, int):
            if not parts:
                parts.append(f"[{segment}]")
            else:
                parts[-1] = f"{parts[-1]}[{segment}]"
            continue
        parts.append(str(segment))
    return ".".join(parts)


def _parse_decimal_number(raw: object, path: Tuple[PathSegment, ...], key: str) -> Decimal:
    if isinstance(raw, bool):
        raise ValueError(
            f"Range spec '{_path_key(path)}' has invalid {key}: bool is not allowed."
        )
    if not isinstance(raw, (int, float, str)):
        raise ValueError(
            f"Range spec '{_path_key(path)}' has invalid {key}: {raw!r}."
        )
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(
            f"Range spec '{_path_key(path)}' has invalid {key}: {raw!r}."
        ) from error


def _expand_numeric_range_spec(
    node: object,
    path: Tuple[PathSegment, ...],
) -> Union[List[object], None]:
    """
    Expand numeric range object to value list.

    Supported form:
    {"min": <number>, "max": <number>, "step": <number>}
    """

    if not isinstance(node, dict):
        return None
    if set(node.keys()) != _RANGE_KEYS:
        return None

    min_raw = node.get("min")
    max_raw = node.get("max")
    step_raw = node.get("step")
    min_value = _parse_decimal_number(min_raw, path, "min")
    max_value = _parse_decimal_number(max_raw, path, "max")
    step_value = _parse_decimal_number(step_raw, path, "step")

    if step_value <= 0:
        raise ValueError(f"Range spec '{_path_key(path)}' requires step > 0.")
    if max_value < min_value:
        raise ValueError(f"Range spec '{_path_key(path)}' requires max >= min.")

    epsilon = step_value / Decimal("1000000000")
    points: List[Decimal] = []
    index = 0
    current = min_value
    while current <= (max_value + epsilon):
        points.append(current)
        index += 1
        if index > _MAX_RANGE_POINTS:
            raise ValueError(
                f"Range spec '{_path_key(path)}' exceeds max points {_MAX_RANGE_POINTS}."
            )
        current = min_value + (step_value * index)

    if points and (points[-1] < (max_value - epsilon)):
        points.append(max_value)
    if not points:
        raise ValueError(f"Range spec '{_path_key(path)}' produced no values.")

    use_integer_values = all(
        isinstance(raw, int) and (not isinstance(raw, bool))
        for raw in (min_raw, max_raw, step_raw)
    )
    if use_integer_values:
        return [int(point) for point in points]
    return [float(point) for point in points]


def _parse_sweep_path(raw: str) -> Tuple[PathSegment, ...]:
    """Parse path syntax like 'worker_class.classes[0].params.a'."""

    path_text = str(raw).strip()
    if not path_text:
        raise ValueError("Sweep variable path must be a non-empty string.")

    segments: List[PathSegment] = []
    parts = path_text.split(".")
    part_pattern = re.compile(r"^([^\[\]]+)((?:\[\d+\])*)$")
    index_pattern = re.compile(r"\[(\d+)\]")

    for part in parts:
        token = part.strip()
        if not token:
            raise ValueError(f"Invalid sweep path: {raw!r}")
        match = part_pattern.match(token)
        if match is None:
            raise ValueError(f"Invalid sweep path segment: {token!r} in {raw!r}")
        key_token = match.group(1)
        segments.append(key_token)
        index_tail = match.group(2)
        if index_tail:
            for index_text in index_pattern.findall(index_tail):
                segments.append(int(index_text))

    return tuple(segments)


def _set_path_value(payload: object, path: Tuple[PathSegment, ...], value: object) -> None:
    """Set value at parsed path, creating missing dict branches."""

    if not path:
        raise ValueError("Sweep path must not be empty.")

    cursor = payload
    for index, segment in enumerate(path[:-1]):
        next_segment = path[index + 1]
        if isinstance(segment, str):
            if not isinstance(cursor, dict):
                raise ValueError(
                    f"Cannot traverse key '{segment}' at '{_path_key(path[: index + 1])}': "
                    "parent is not an object."
                )
            if segment not in cursor or cursor[segment] is None:
                cursor[segment] = [] if isinstance(next_segment, int) else {}
            elif isinstance(next_segment, int) and not isinstance(cursor[segment], list):
                raise ValueError(
                    f"Path '{_path_key(path[: index + 2])}' expects a list."
                )
            elif isinstance(next_segment, str) and not isinstance(cursor[segment], dict):
                raise ValueError(
                    f"Path '{_path_key(path[: index + 2])}' expects an object."
                )
            cursor = cursor[segment]
            continue

        if not isinstance(cursor, list):
            raise ValueError(
                f"Cannot traverse index [{segment}] at '{_path_key(path[: index + 1])}': "
                "parent is not a list."
            )
        if segment < 0 or segment >= len(cursor):
            raise ValueError(
                f"Index out of range at '{_path_key(path[: index + 1])}': {segment}."
            )
        if cursor[segment] is None:
            cursor[segment] = [] if isinstance(next_segment, int) else {}
        elif isinstance(next_segment, int) and not isinstance(cursor[segment], list):
            raise ValueError(
                f"Path '{_path_key(path[: index + 2])}' expects a list."
            )
        elif isinstance(next_segment, str) and not isinstance(cursor[segment], dict):
            raise ValueError(
                f"Path '{_path_key(path[: index + 2])}' expects an object."
            )
        cursor = cursor[segment]

    final_segment = path[-1]
    if isinstance(final_segment, str):
        if not isinstance(cursor, dict):
            raise ValueError(
                f"Cannot set key '{final_segment}' at '{_path_key(path)}': "
                "parent is not an object."
            )
        cursor[final_segment] = copy.deepcopy(value)
        return

    if not isinstance(cursor, list):
        raise ValueError(
            f"Cannot set index [{final_segment}] at '{_path_key(path)}': "
            "parent is not a list."
        )
    if final_segment < 0 or final_segment >= len(cursor):
        raise ValueError(
            f"Index out of range at '{_path_key(path)}': {final_segment}."
        )
    cursor[final_segment] = copy.deepcopy(value)


def _expand_sweep_values(raw_values: object, path: Tuple[PathSegment, ...]) -> List[object]:
    """Expand one variable's sweep values."""

    expanded_range = _expand_numeric_range_spec(raw_values, path)
    if expanded_range is not None:
        return [copy.deepcopy(item) for item in expanded_range]

    if not isinstance(raw_values, list):
        raise ValueError(
            f"Sweep variable '{_path_key(path)}' must be a list or range object {{min,max,step}}."
        )
    if not raw_values:
        raise ValueError(f"Sweep variable '{_path_key(path)}' must not be an empty list.")
    return [copy.deepcopy(item) for item in raw_values]


def _normalize_payload_for_controller_mode(
    payload: Dict[str, object],
    controller_mode: str,
    policy: str,
) -> Dict[str, object]:
    """Apply policy-specific controller filtering to one simulation payload."""

    normalized = copy.deepcopy(payload)
    if controller_mode == "none":
        normalized.pop("controller", None)
        return normalized

    controller_payload = normalized.get("controller")
    if (controller_payload is not None) and (not isinstance(controller_payload, dict)):
        raise ValueError("'controller' in simulation payload must be a JSON object.")

    if controller_mode == "latency_only" and isinstance(controller_payload, dict):
        latency_payload = controller_payload.get("latency_tracker")
        if latency_payload is None:
            normalized["controller"] = {}
        else:
            normalized["controller"] = {
                "latency_tracker": copy.deepcopy(latency_payload)
            }
        return normalized

    if controller_mode == "full" and isinstance(controller_payload, dict):
        if policy == "lp-wrr":
            normalized["controller"] = {
                key: copy.deepcopy(value)
                for key, value in controller_payload.items()
                if key in {"latency_tracker", "lp-wrr"}
            }
        elif policy == "sp-wrr":
            normalized["controller"] = {
                key: copy.deepcopy(value)
                for key, value in controller_payload.items()
                if key in {"latency_tracker", "sp-wrr"}
            }

    return normalized


def _filter_sweep_spec_for_controller_mode(
    sweep_spec: Dict[str, object],
    controller_mode: str,
    policy: str,
) -> Dict[str, object]:
    """Drop irrelevant controller sweep variables for each policy mode."""

    filtered: Dict[str, object] = {}
    for path_text, raw_values in sweep_spec.items():
        parsed_path = _parse_sweep_path(path_text)
        if parsed_path and parsed_path[0] == "controller":
            if controller_mode == "none":
                continue
            if controller_mode == "latency_only":
                if len(parsed_path) < 2 or parsed_path[1] != "latency_tracker":
                    continue
            if controller_mode == "full":
                if len(parsed_path) < 2:
                    continue
                if policy == "lp-wrr" and parsed_path[1] not in {"latency_tracker", "lp-wrr"}:
                    continue
                if policy == "sp-wrr" and parsed_path[1] not in {"latency_tracker", "sp-wrr"}:
                    continue
        filtered[path_text] = raw_values
    return filtered


def _parse_policy_list(raw: str) -> List[str]:
    tokens = [token.strip() for token in str(raw).split(",")]
    normalized: List[str] = []
    seen = set()
    for token in tokens:
        if not token:
            continue
        canonical = normalize_policy_name(token)
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    if not normalized:
        raise ValueError("Policy list must contain at least one non-empty policy name.")
    return normalized


def _parse_sweep_plan(path: Path) -> Tuple[Dict[str, object], List[ScenarioDefinition], Dict[str, object]]:
    """Parse new sweep plan schema with base_config, scenarios, and sweeps."""

    payload = _load_json_object(path)

    base_config_raw = payload.get("base_config")
    if not isinstance(base_config_raw, dict):
        raise ValueError("Sweep config must contain object key 'base_config'.")

    scenarios_raw = payload.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("Sweep config must contain non-empty list key 'scenarios'.")

    sweeps_raw = payload.get("sweeps", {})
    if sweeps_raw is None:
        sweeps_raw = {}
    if not isinstance(sweeps_raw, dict):
        raise ValueError("'sweeps' must be an object mapping scenario_name -> sweep variables.")

    scenario_names = set()
    scenarios: List[ScenarioDefinition] = []
    for idx, raw_scenario in enumerate(scenarios_raw):
        if not isinstance(raw_scenario, dict):
            raise ValueError(f"scenarios[{idx}] must be an object.")

        name_raw = raw_scenario.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            raise ValueError(f"scenarios[{idx}].name must be a non-empty string.")
        scenario_name = name_raw.strip()
        if scenario_name in scenario_names:
            raise ValueError(f"Duplicate scenario name: {scenario_name!r}.")
        scenario_names.add(scenario_name)

        override_raw = raw_scenario.get("override", {})
        if override_raw is None:
            override_raw = {}
        if not isinstance(override_raw, dict):
            raise ValueError(f"scenarios[{idx}].override must be an object.")

        sweep_spec_raw = sweeps_raw.get(scenario_name, {})
        if sweep_spec_raw is None:
            sweep_spec_raw = {}
        if not isinstance(sweep_spec_raw, dict):
            raise ValueError(
                f"sweeps[{scenario_name!r}] must be an object mapping path -> values."
            )

        scenarios.append(
            ScenarioDefinition(
                name=scenario_name,
                override=copy.deepcopy(override_raw),
                sweep_spec=copy.deepcopy(sweep_spec_raw),
            )
        )

    unknown_sweep_names = sorted(set(sweeps_raw.keys()) - scenario_names)
    if unknown_sweep_names:
        raise ValueError(
            "'sweeps' contains unknown scenarios: " + ", ".join(unknown_sweep_names)
        )

    return copy.deepcopy(base_config_raw), scenarios, payload


def _build_cases_for_scenario(
    base_config: Dict[str, object],
    scenario: ScenarioDefinition,
    config_dir: Path,
    controller_mode: str,
    policy: str,
    case_id_start: int,
) -> Tuple[List[SweepCase], int]:
    """Build expanded cases for one scenario under one policy controller mode."""

    scenario_payload = _merge_dicts_deep(base_config, scenario.override)
    filtered_sweep_spec = _filter_sweep_spec_for_controller_mode(
        scenario.sweep_spec,
        controller_mode=controller_mode,
        policy=policy,
    )

    variables: List[Tuple[str, Tuple[PathSegment, ...], List[object]]] = []
    for path_text, raw_values in filtered_sweep_spec.items():
        parsed_path = _parse_sweep_path(path_text)
        expanded_values = _expand_sweep_values(raw_values, parsed_path)
        variables.append((path_text, parsed_path, expanded_values))

    combinations: List[List[Tuple[str, Tuple[PathSegment, ...], object]]] = [[]]
    for path_text, parsed_path, values in variables:
        next_combinations: List[List[Tuple[str, Tuple[PathSegment, ...], object]]] = []
        for base_combo in combinations:
            for value in values:
                next_combo = list(base_combo)
                next_combo.append((path_text, parsed_path, copy.deepcopy(value)))
                next_combinations.append(next_combo)
        combinations = next_combinations

    cases: List[SweepCase] = []
    next_case_id = case_id_start
    for scenario_case_id, combo in enumerate(combinations, start=1):
        simulation_payload = copy.deepcopy(scenario_payload)
        selected_values: Dict[str, object] = {}
        for path_text, parsed_path, value in combo:
            _set_path_value(simulation_payload, parsed_path, value)
            selected_values[path_text] = copy.deepcopy(value)

        normalized_payload = _normalize_payload_for_controller_mode(
            simulation_payload,
            controller_mode=controller_mode,
            policy=policy,
        )
        normalized_payload = _resolve_service_trace_paths(
            normalized_payload,
            config_dir=config_dir,
        )
        _validate_service_topology_schema(normalized_payload)

        cases.append(
            SweepCase(
                case_id=next_case_id,
                scenario_name=scenario.name,
                scenario_case_id=scenario_case_id,
                simulation_payload=normalized_payload,
                selected_values=selected_values,
            )
        )
        next_case_id += 1

    return cases, next_case_id


def _build_cases(
    base_config: Dict[str, object],
    scenarios: Sequence[ScenarioDefinition],
    config_dir: Path,
    controller_mode: str,
    policy: str,
) -> List[SweepCase]:
    """Build all scenarios for one controller mode."""

    all_cases: List[SweepCase] = []
    next_case_id = 1
    for scenario in scenarios:
        scenario_cases, next_case_id = _build_cases_for_scenario(
            base_config=base_config,
            scenario=scenario,
            config_dir=config_dir,
            controller_mode=controller_mode,
            policy=policy,
            case_id_start=next_case_id,
        )
        all_cases.extend(scenario_cases)
    return all_cases


def _json_compact(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return str(value)


def _metric_entity_sort_key(raw: object) -> Tuple[int, object]:
    text = str(raw)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _extract_summary_metric_columns(summary: Dict[str, object]) -> Dict[str, object]:
    """Flatten summary outcomes + breakdown metrics to CSV columns."""

    out: Dict[str, object] = {
        "outcomes.traffic.dispatched": summary.get("dispatched", 0),
        "outcomes.traffic.completed": summary.get("completed", 0),
        "outcomes.traffic.dropped": summary.get("dropped", 0),
        "outcomes.traffic.drop_rate": summary.get("drop_rate", 0.0),
        "outcomes.traffic.throughput": summary.get("throughput", 0.0),
        "outcomes.latency.mean": summary.get("mean_latency", 0.0),
        "outcomes.latency.median": summary.get("median_latency", 0.0),
        "outcomes.latency.p95": summary.get("p95_latency", 0.0),
        "outcomes.latency.p99": summary.get("p99_latency", 0.0),
        "outcomes.queueing.avg_queue_len": summary.get("avg_queue_len", 0.0),
        "outcomes.queueing.avg_global_inflight": summary.get("avg_global_inflight", 0.0),
        "outcomes.queueing.avg_utilization": summary.get("avg_utilization", 0.0),
        "outcomes.dispersion.worker_latency_mean_stddev": summary.get(
            "worker_latency_mean_stddev", 0.0
        ),
        "outcomes.dispersion.worker_latency_mean_max_gap": summary.get(
            "worker_latency_mean_max_gap", 0.0
        ),
        "outcomes.dispersion.service_latency_mean_stddev": summary.get(
            "service_latency_mean_stddev", 0.0
        ),
        "outcomes.dispersion.service_latency_mean_max_gap": summary.get(
            "service_latency_mean_max_gap", 0.0
        ),
        "outcomes.dispersion.worker_utilization_stddev": summary.get(
            "worker_utilization_stddev", 0.0
        ),
        "outcomes.dispersion.worker_utilization_max_gap": summary.get(
            "worker_utilization_max_gap", 0.0
        ),
        "outcomes.job_size.total": summary.get("total_job_size", 0),
        "outcomes.time.sim_time_total": summary.get("sim_time_total", 0.0),
        "outcomes.time.wall_time_total": summary.get("wall_time_total", 0.0),
        "outcomes.time.drain_time": summary.get("drain_time", 0.0),
    }

    job_size_by_class = summary.get("total_job_size_by_class", {})
    if isinstance(job_size_by_class, dict):
        for class_id, total in sorted(
            job_size_by_class.items(), key=lambda item: _metric_entity_sort_key(item[0])
        ):
            out[f"breakdown.service.job_size.class_{class_id}.total"] = total

    latency_by_class = summary.get("latency_by_class", {})
    if isinstance(latency_by_class, dict):
        for class_id, stats in sorted(
            latency_by_class.items(), key=lambda item: _metric_entity_sort_key(item[0])
        ):
            if not isinstance(stats, dict):
                continue
            for metric_key in ("count", "mean", "median", "p95", "p99"):
                out[
                    f"breakdown.service.latency.class_{class_id}.{metric_key}"
                ] = stats.get(metric_key, 0)

    latency_by_worker = summary.get("latency_by_worker", {})
    if isinstance(latency_by_worker, dict):
        for worker_id, stats in sorted(
            latency_by_worker.items(), key=lambda item: _metric_entity_sort_key(item[0])
        ):
            if not isinstance(stats, dict):
                continue
            for metric_key in ("count", "mean", "median", "p95", "p99"):
                out[
                    f"breakdown.worker.latency.worker_{worker_id}.{metric_key}"
                ] = stats.get(metric_key, 0)

    drop_by_class = summary.get("drop_by_class", {})
    if isinstance(drop_by_class, dict):
        for class_id, stats in sorted(
            drop_by_class.items(), key=lambda item: _metric_entity_sort_key(item[0])
        ):
            if not isinstance(stats, dict):
                continue
            for metric_key in ("dispatched", "dropped", "drop_rate"):
                out[
                    f"breakdown.service.drop.class_{class_id}.{metric_key}"
                ] = stats.get(metric_key, 0)

    drop_by_worker = summary.get("drop_by_worker", {})
    if isinstance(drop_by_worker, dict):
        for worker_id, stats in sorted(
            drop_by_worker.items(), key=lambda item: _metric_entity_sort_key(item[0])
        ):
            if not isinstance(stats, dict):
                continue
            for metric_key in ("dispatched", "dropped", "drop_rate"):
                out[
                    f"breakdown.worker.drop.worker_{worker_id}.{metric_key}"
                ] = stats.get(metric_key, 0)

    utilization_by_worker = summary.get("utilization_by_worker", [])
    if isinstance(utilization_by_worker, list):
        for worker_id, utilization in enumerate(utilization_by_worker):
            out[f"breakdown.worker.utilization.worker_{worker_id}"] = utilization

    return out


def _write_results_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    selector_keys: Sequence[str],
) -> None:
    fixed_fields = [
        "run_id",
        "case_id",
        "scenario",
        "scenario_case_id",
        "status",
        "policy",
        "controller_sweep",
        "run_dir",
        "mean_latency",
        "median_latency",
        "p95_latency",
        "p99_latency",
        "error",
    ]
    fixed_set = set(fixed_fields)
    selector_keys_sorted = sorted(set(selector_keys))
    selector_key_set = set(selector_keys_sorted)
    metric_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if (key not in fixed_set) and (key not in selector_key_set)
        }
    )
    fields = fixed_fields[:-1] + selector_keys_sorted + metric_keys + ["error"]

    normalized_rows: List[Dict[str, object]] = []
    for row in rows:
        out: Dict[str, object] = {}
        for field in fields:
            raw_value = row.get(field, "")
            if field in selector_key_set:
                out[field] = _json_compact(raw_value)
            else:
                out[field] = raw_value
        normalized_rows.append(out)

    result_df = pd.DataFrame(normalized_rows, columns=fields)
    result_df.to_csv(path, index=False)


def _load_existing_results(path: Path) -> List[Dict[str, object]]:
    """Load existing sweep rows from CSV for resume mode."""

    if not path.exists():
        return []
    result_df = pd.read_csv(path, keep_default_na=False)
    return result_df.to_dict(orient="records")


def _row_case_key(row: Dict[str, object]) -> Optional[Tuple[str, int]]:
    """Build (policy, case_id) key from one result row."""

    policy = str(row.get("policy", "")).strip().lower()
    if not policy:
        return None
    raw_case_id = row.get("case_id", "")
    try:
        case_id = int(str(raw_case_id).strip())
    except (TypeError, ValueError):
        return None
    return policy, case_id


def _row_run_id(row: Dict[str, object]) -> int:
    """Extract run_id from one result row; return 0 on parse failure."""

    raw_run_id = row.get("run_id", "")
    try:
        return int(str(raw_run_id).strip())
    except (TypeError, ValueError):
        return 0


def _persist_sweep_artifacts(
    report_dir: Path,
    output_csv: Path,
    raw_input_payload: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    selector_keys: Sequence[str],
) -> Tuple[Path, Path]:
    """Persist sweep CSV/JSON and input snapshot."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_results_csv(output_csv, rows, selector_keys=selector_keys)

    summary_json = report_dir / "controller_sweep_results.json"
    summary_json.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot_path = report_dir / "input_config.snapshot.json"
    snapshot_path.write_text(
        json.dumps(raw_input_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_json, snapshot_path


def _build_run_prefix(policy: str, scenario_name: str, case_id: int) -> str:
    """Build a collision-resistant run prefix for one sweep case."""

    policy_token = _sanitize_name(policy)
    scenario_token = _sanitize_name(scenario_name)
    return f"run-sweep-{policy_token}-{scenario_token}-case{int(case_id):04d}"


def _build_case_label(
    policy: str,
    controller_mode: str,
    case_id: int,
    scenario_name: str,
    scenario_case_id: int,
    selected_values: Dict[str, object],
) -> str:
    """Build human-readable sweep case label for console logs."""

    selected_text = {
        key: _json_compact(value) for key, value in selected_values.items()
    }
    return (
        f"case={case_id} scenario={scenario_name} scenario_case={scenario_case_id} "
        f"policy={policy} controller_mode={controller_mode} sweep={selected_text}"
    )


def _execute_case_job(
    *,
    run_id: int,
    case_id: int,
    scenario_name: str,
    scenario_case_id: int,
    policy: str,
    controller_sweep: bool,
    selected_values: Dict[str, object],
    case_cfg_path: str,
    t_end_seconds: float,
    seed: int,
    detail: bool,
    logger_mode: str,
    run_prefix: str,
    logs_root: str,
) -> Dict[str, object]:
    """
    Execute one sweep case and return one result row.

    This function is process-safe and used by both sequential and parallel modes.
    """

    try:
        summary = run_simulation(
            t_end=t_end_seconds,
            policy=policy,
            simulation_config=Path(case_cfg_path),
            seed=seed,
            detail=detail,
            logger_mode=logger_mode,
            run_prefix=run_prefix,
            logs_root=Path(logs_root),
        )
        row: Dict[str, object] = {
            "run_id": run_id,
            "case_id": case_id,
            "scenario": scenario_name,
            "scenario_case_id": scenario_case_id,
            "status": "ok",
            "policy": policy,
            "controller_sweep": controller_sweep,
            "run_dir": summary.get("run_dir", ""),
            "mean_latency": summary.get("mean_latency", 0.0),
            "median_latency": summary.get("median_latency", 0.0),
            "p95_latency": summary.get("p95_latency", 0.0),
            "p99_latency": summary.get("p99_latency", 0.0),
            "error": "",
        }
        row.update(selected_values)
        row.update(_extract_summary_metric_columns(summary))
        return row
    except Exception as error:  # noqa: BLE001
        row = {
            "run_id": run_id,
            "case_id": case_id,
            "scenario": scenario_name,
            "scenario_case_id": scenario_case_id,
            "status": "failed",
            "policy": policy,
            "controller_sweep": controller_sweep,
            "run_dir": "",
            "mean_latency": 0.0,
            "median_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "error": str(error),
        }
        row.update(selected_values)
        row.update(_extract_summary_metric_columns({}))
        return row


def _resolve_jobs(raw_jobs: int) -> int:
    """Resolve CLI jobs value to an effective worker count."""

    jobs = int(raw_jobs)
    if jobs < 0:
        raise ValueError("--jobs must be >= 0.")
    if jobs == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count)
    return max(1, jobs)


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return sanitized or "scenario"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run simulation sweeps from a sweep plan "
            "(base_config + scenarios + per-scenario sweeps)."
        )
    )
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
        type=str,
        default="static-wrr",
        help=(
            "Load balancing policy or comma-separated list "
            "(aliases supported: swrr,lpwrr,spwrr,lc,p2c,mema,lp2c,...)"
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
            "Sweep config JSON with top-level keys: "
            "base_config, scenarios, sweeps."
        ),
    )
    parser.add_argument("-S", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help=(
            "Parallel worker processes. "
            "Use 1 for sequential (default), 0 for all CPU cores."
        ),
    )
    parser.add_argument(
        "-d",
        "--detail",
        action="store_true",
        help="Write per-request detail metrics to CSV for each run",
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
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report folder for sweep artifacts (default: logs/sweep-YYYYMMDD-HHMMSS).",
    )
    parser.add_argument(
        "--resume-report-dir",
        type=Path,
        default=None,
        help=(
            "Resume an unfinished sweep from an existing report folder. "
            "Cases already recorded in controller_sweep_results.csv are skipped."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <report_dir>/controller_sweep_results.csv).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs if one run fails.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        t_end_seconds = parse_duration_seconds(args.t_end)
        policies = _parse_policy_list(args.policy)
        logger_mode = normalize_log_mode(args.logger_mode)
        jobs = _resolve_jobs(args.jobs)
        base_config, scenarios, raw_input_payload = _parse_sweep_plan(args.simulation_config)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if (args.report_dir is not None) and (args.resume_report_dir is not None):
        raise SystemExit("Use either --report-dir or --resume-report-dir, not both.")

    resume_mode = args.resume_report_dir is not None
    if resume_mode:
        report_dir = args.resume_report_dir
        if report_dir is None or (not report_dir.exists()) or (not report_dir.is_dir()):
            raise SystemExit(
                f"Resume report folder not found or not a directory: {args.resume_report_dir}"
            )
    else:
        report_dir = args.report_dir
        if report_dir is None:
            stamp = datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
            report_dir = Path("logs") / stamp
        report_dir.mkdir(parents=True, exist_ok=False)

    if args.output_csv is None:
        output_csv = report_dir / "controller_sweep_results.csv"
    else:
        output_csv = (
            args.output_csv if args.output_csv.is_absolute() else (report_dir / args.output_csv)
        )

    snapshot_path = report_dir / "input_config.snapshot.json"
    if resume_mode and snapshot_path.exists():
        try:
            existing_snapshot = _load_json_object(snapshot_path)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if existing_snapshot != raw_input_payload:
            raise SystemExit(
                "Resume folder was created from a different sweep config. "
                "Use the same -f/--config as the original run."
            )

    configs_dir = report_dir / "case_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = report_dir / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    config_dir = args.simulation_config.resolve().parent
    policy_cases: Dict[str, List[SweepCase]] = {}
    total_runs = 0
    for policy in policies:
        controller_mode = policy_controller_mode(policy)
        try:
            cases = _build_cases(
                base_config=base_config,
                scenarios=scenarios,
                config_dir=config_dir,
                controller_mode=controller_mode,
                policy=policy,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

        if controller_mode in {"latency_only", "full"}:
            required_block = "lp-wrr" if policy == "lp-wrr" else "sp-wrr"
            for case in cases:
                controller_payload = case.simulation_payload.get("controller")
                if not isinstance(controller_payload, dict):
                    raise SystemExit(
                        (
                            f"Policy '{policy}' requires controller config, but scenario "
                            f"'{case.scenario_name}' case {case.scenario_case_id} has no "
                            "controller block."
                        )
                    )
                latency_tracker_payload = controller_payload.get("latency_tracker")
                if not isinstance(latency_tracker_payload, dict):
                    raise SystemExit(
                        (
                            f"Policy '{policy}' requires controller.latency_tracker, but scenario "
                            f"'{case.scenario_name}' case {case.scenario_case_id} is missing it."
                        )
                    )
                if controller_mode == "full":
                    policy_payload = controller_payload.get(required_block)
                    if not isinstance(policy_payload, dict):
                        raise SystemExit(
                            (
                                f"Policy '{policy}' requires controller.{required_block}, but "
                                f"scenario '{case.scenario_name}' case {case.scenario_case_id} "
                                "is missing it."
                            )
                        )

        policy_cases[policy] = cases
        total_runs += len(cases)
        print(
            f"Policy {policy}: {len(cases)} cases "
            f"(controller mode={controller_mode})"
        )

    selector_keys_seen: set[str] = set()
    for cases in policy_cases.values():
        for case in cases:
            selector_keys_seen.update(case.selected_values.keys())
    selector_keys_sorted = sorted(selector_keys_seen)

    results: List[Dict[str, object]] = []
    completed_case_keys: set[Tuple[str, int]] = set()
    run_id = 0
    if resume_mode:
        try:
            existing_rows = _load_existing_results(output_csv)
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"Failed to load existing CSV for resume: {error}") from error
        results.extend(existing_rows)
        for row in existing_rows:
            case_key = _row_case_key(row)
            if case_key is not None:
                completed_case_keys.add(case_key)
            run_id = max(run_id, _row_run_id(row))
        print(
            "Resume mode: "
            f"loaded {len(existing_rows)} existing row(s), "
            f"{len(completed_case_keys)} unique completed case key(s)."
        )

    completed_case_count = 0
    for policy, cases in policy_cases.items():
        for case in cases:
            if (policy, case.case_id) in completed_case_keys:
                completed_case_count += 1
    pending_runs = total_runs - completed_case_count
    print(
        f"Total simulation runs: {total_runs} "
        f"(pending={pending_runs}, completed={completed_case_count})"
    )

    _persist_sweep_artifacts(
        report_dir=report_dir,
        output_csv=output_csv,
        raw_input_payload=raw_input_payload,
        rows=results,
        selector_keys=selector_keys_sorted,
    )

    pending_tasks: List[Dict[str, object]] = []
    pending_index = 0
    for policy in policies:
        controller_mode = policy_controller_mode(policy)
        use_controller_sweep = controller_mode != "none"
        policy_case_dir = configs_dir / policy
        policy_case_dir.mkdir(parents=True, exist_ok=True)
        cases = policy_cases[policy]
        for case in cases:
            case_key = (policy, case.case_id)
            if case_key in completed_case_keys:
                print(
                    f"[skip] case={case.case_id} scenario={case.scenario_name} "
                    f"scenario_case={case.scenario_case_id} policy={policy} (already recorded)"
                )
                continue

            pending_index += 1
            run_id += 1
            scenario_dir = policy_case_dir / _sanitize_name(case.scenario_name)
            scenario_dir.mkdir(parents=True, exist_ok=True)
            case_cfg_path = scenario_dir / f"simulation_case_{case.scenario_case_id:04d}.json"
            case_cfg_path.write_text(
                json.dumps(case.simulation_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            label = _build_case_label(
                policy=policy,
                controller_mode=controller_mode,
                case_id=case.case_id,
                scenario_name=case.scenario_name,
                scenario_case_id=case.scenario_case_id,
                selected_values=case.selected_values,
            )
            print(f"[queue {pending_index}/{pending_runs}] {label}")
            pending_tasks.append(
                {
                    "pending_index": pending_index,
                    "label": label,
                    "job_args": {
                        "run_id": run_id,
                        "case_id": case.case_id,
                        "scenario_name": case.scenario_name,
                        "scenario_case_id": case.scenario_case_id,
                        "policy": policy,
                        "controller_sweep": use_controller_sweep,
                        "selected_values": copy.deepcopy(case.selected_values),
                        "case_cfg_path": str(case_cfg_path),
                        "t_end_seconds": t_end_seconds,
                        "seed": args.seed,
                        "detail": args.detail,
                        "logger_mode": logger_mode,
                        "run_prefix": _build_run_prefix(
                            policy=policy,
                            scenario_name=case.scenario_name,
                            case_id=case.case_id,
                        ),
                        "logs_root": str(run_logs_dir),
                    },
                }
            )

    stop_due_to_error = False
    if jobs == 1:
        for task in pending_tasks:
            pending_pos = int(task["pending_index"])
            label = str(task["label"])
            job_args = dict(task["job_args"])
            print(f"[run {pending_pos}/{pending_runs}] {label}")
            row = _execute_case_job(**job_args)
            results.append(row)
            if str(row.get("status", "")).strip().lower() == "failed":
                print(f"  -> FAILED: {row.get('error', '')}")
                if not args.continue_on_error:
                    stop_due_to_error = True

            summary_json, snapshot_path = _persist_sweep_artifacts(
                report_dir=report_dir,
                output_csv=output_csv,
                raw_input_payload=raw_input_payload,
                rows=results,
                selector_keys=selector_keys_sorted,
            )
            if stop_due_to_error:
                break
    elif pending_tasks:
        print(f"Parallel execution enabled: workers={jobs}")
        try:
            completed_parallel = 0
            task_iter = iter(pending_tasks)
            future_to_task = {}
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                while len(future_to_task) < jobs:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        break
                    future = executor.submit(
                        _execute_case_job,
                        **dict(task["job_args"]),
                    )
                    future_to_task[future] = task

                while future_to_task:
                    done, _ = wait(
                        set(future_to_task.keys()),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        task = future_to_task.pop(future)
                        completed_parallel += 1
                        label = str(task["label"])
                        job_args = dict(task["job_args"])
                        try:
                            row = future.result()
                        except Exception as error:  # noqa: BLE001
                            row = {
                                "run_id": job_args["run_id"],
                                "case_id": job_args["case_id"],
                                "scenario": job_args["scenario_name"],
                                "scenario_case_id": job_args["scenario_case_id"],
                                "status": "failed",
                                "policy": job_args["policy"],
                                "controller_sweep": job_args["controller_sweep"],
                                "run_dir": "",
                                "mean_latency": 0.0,
                                "median_latency": 0.0,
                                "p95_latency": 0.0,
                                "p99_latency": 0.0,
                                "error": str(error),
                            }
                            row.update(dict(job_args["selected_values"]))
                            row.update(_extract_summary_metric_columns({}))

                        results.append(row)
                        status = str(row.get("status", "")).strip().lower()
                        print(
                            f"[done {completed_parallel}/{pending_runs}] {label} "
                            f"status={status or 'unknown'}"
                        )
                        if status == "failed":
                            print(f"  -> FAILED: {row.get('error', '')}")
                            if not args.continue_on_error:
                                stop_due_to_error = True

                        summary_json, snapshot_path = _persist_sweep_artifacts(
                            report_dir=report_dir,
                            output_csv=output_csv,
                            raw_input_payload=raw_input_payload,
                            rows=results,
                            selector_keys=selector_keys_sorted,
                        )

                    if stop_due_to_error:
                        continue
                    while len(future_to_task) < jobs:
                        try:
                            task = next(task_iter)
                        except StopIteration:
                            break
                        future = executor.submit(
                            _execute_case_job,
                            **dict(task["job_args"]),
                        )
                        future_to_task[future] = task
        except (PermissionError, OSError) as error:
            print(
                "Parallel execution is unavailable in this environment; "
                f"falling back to sequential mode. reason={error}"
            )
            completed_run_ids = {_row_run_id(row) for row in results}
            for task in pending_tasks:
                job_args = dict(task["job_args"])
                run_id_value = int(job_args["run_id"])
                if run_id_value in completed_run_ids:
                    continue
                pending_pos = int(task["pending_index"])
                label = str(task["label"])
                print(f"[run-fallback {pending_pos}/{pending_runs}] {label}")
                row = _execute_case_job(**job_args)
                results.append(row)
                completed_run_ids.add(run_id_value)
                if str(row.get("status", "")).strip().lower() == "failed":
                    print(f"  -> FAILED: {row.get('error', '')}")
                    if not args.continue_on_error:
                        stop_due_to_error = True

                summary_json, snapshot_path = _persist_sweep_artifacts(
                    report_dir=report_dir,
                    output_csv=output_csv,
                    raw_input_payload=raw_input_payload,
                    rows=results,
                    selector_keys=selector_keys_sorted,
                )
                if stop_due_to_error:
                    break

    summary_json, snapshot_path = _persist_sweep_artifacts(
        report_dir=report_dir,
        output_csv=output_csv,
        raw_input_payload=raw_input_payload,
        rows=results,
        selector_keys=selector_keys_sorted,
    )

    ok_count = sum(1 for row in results if row.get("status") == "ok")
    failed_count = len(results) - ok_count
    print(f"Sweep completed: total={len(results)} ok={ok_count} failed={failed_count}")
    print(f"Run logs dir: {run_logs_dir}")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {summary_json}")
    print(f"Saved config snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()

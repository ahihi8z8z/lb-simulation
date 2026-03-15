#!/usr/bin/env python3
"""Compare latency metrics across multiple simulator log folders."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as error:
    raise SystemExit(
        "matplotlib and pandas are required for this tool. "
        "Install with: pip install -r tools/requirements.txt"
    ) from error


METRICS: Tuple[str, ...] = ("mean", "median", "p95", "p99")
_METRIC_TITLE = {
    "mean": "Mean",
    "median": "Median",
    "p95": "P95",
    "p99": "P99",
}


@dataclass
class RunMetrics:
    """Loaded and normalized metrics for one run label."""

    run_dir: Path
    label: str
    system: Dict[str, float]
    service: Dict[str, Dict[str, float]]
    worker: Dict[str, Dict[str, float]]
    service_config_canonical: str
    worker_config_canonical: str
    service_ids: Tuple[str, ...]
    worker_ids: Tuple[str, ...]


def _class_sort_key(raw: str) -> Tuple[int, object]:
    try:
        return (0, int(raw))
    except ValueError:
        return (1, raw)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (p / 100.0)
    left = int(pos)
    right = min(left + 1, len(sorted_vals) - 1)
    weight = pos - left
    return sorted_vals[left] * (1.0 - weight) + sorted_vals[right] * weight


def _stats(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
    }


def _canonical_json(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _summary_entity_has_all_metrics(entity: Mapping[str, object]) -> bool:
    return all(metric in entity for metric in METRICS)


def _load_json_dict(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _read_nested_mapping(
    source: Mapping[str, object],
    *path: str,
) -> Dict[str, object]:
    current: object = source
    for key in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return dict(current) if isinstance(current, Mapping) else {}


def _infer_label(
    run_dir: Path,
    summary: Mapping[str, object],
    run_config: Mapping[str, object],
) -> str:
    policy_raw = summary.get("policy")
    if not policy_raw:
        policy_raw = _read_nested_mapping(summary, "meta").get("policy", "")
    if not policy_raw:
        policy_raw = run_config.get("policy", "")
    policy = str(policy_raw).strip().lower()
    if not policy:
        return run_dir.name

    return policy


def _load_detail_stats(
    detail_csv: Path,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    detail_df = pd.read_csv(detail_csv)
    required = {"latency", "class_id", "worker_id"}
    if not required.issubset(set(detail_df.columns)):
        raise ValueError(f"{detail_csv} must contain columns: {sorted(required)}.")

    detail_df = detail_df.copy()
    detail_df["latency"] = pd.to_numeric(detail_df["latency"], errors="coerce")
    detail_df = detail_df.dropna(subset=["latency"])
    if detail_df.empty:
        return _stats([]), {}, {}

    detail_df["_class_key"] = (
        detail_df["class_id"]
        .astype(str)
        .str.strip()
        .replace("", "unknown")
        .fillna("unknown")
    )
    detail_df["_worker_key"] = (
        detail_df["worker_id"]
        .astype(str)
        .str.strip()
        .replace("", "unknown")
        .fillna("unknown")
    )

    total_latencies = detail_df["latency"].tolist()
    latencies_by_service = {
        str(class_id): group["latency"].tolist()
        for class_id, group in detail_df.groupby("_class_key", sort=False)
    }
    latencies_by_worker = {
        str(worker_id): group["latency"].tolist()
        for worker_id, group in detail_df.groupby("_worker_key", sort=False)
    }
    return (
        _stats(total_latencies),
        {key: _stats(values) for key, values in latencies_by_service.items()},
        {key: _stats(values) for key, values in latencies_by_worker.items()},
    )


def _resolve_detail_csv(run_dir: Path, summary: Mapping[str, object]) -> Path | None:
    summary_detail_path = str(summary.get("detail_metrics_file", "")).strip()
    if not summary_detail_path:
        artifacts = _read_nested_mapping(summary, "artifacts")
        summary_detail_path = str(artifacts.get("detail_metrics_file", "")).strip()
    candidates: List[Path] = []
    if summary_detail_path:
        detail_path = Path(summary_detail_path)
        candidates.append(detail_path if detail_path.is_absolute() else detail_path)
    candidates.append(run_dir / "request_detail_metrics.csv")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _coerce_metric_block(
    source: Mapping[str, object] | None,
) -> Dict[str, float] | None:
    if source is None:
        return None
    out: Dict[str, float] = {}
    for metric in METRICS:
        raw_value = source.get(metric)
        if raw_value is None:
            return None
        out[metric] = float(raw_value)
    return out


def _expand_run_dirs(raw_dir_expr: str) -> List[Path]:
    expr = raw_dir_expr.strip()
    if not expr:
        return []

    if glob.has_magic(expr):
        matched_paths = sorted(Path(value) for value in glob.glob(expr))
        matched_dirs = [path for path in matched_paths if path.is_dir()]
        if not matched_dirs:
            raise ValueError(
                f"Pattern '{raw_dir_expr}' did not match any log directories."
            )
        return matched_dirs

    run_dir = Path(expr)
    if not run_dir.is_dir():
        raise ValueError(f"Log directory does not exist: {run_dir}")
    return [run_dir]


def _load_run_metrics(run_dir: Path, label_override: Optional[str]) -> RunMetrics:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"Missing summary file: {summary_path}")
    summary = _load_json_dict(summary_path)
    if not summary:
        raise ValueError(f"Invalid summary format: {summary_path}")
    run_config = _load_json_dict(run_dir / "run_config.json")
    if not run_config:
        run_config = _read_nested_mapping(summary, "configs", "run")

    service_cfg_path = run_dir / "service_class_config.json"
    worker_cfg_path = run_dir / "worker_class_config.json"
    if not service_cfg_path.exists() or not worker_cfg_path.exists():
        raise ValueError(
            f"{run_dir} is missing service/worker snapshot config; cannot compare runs."
        )

    outcomes = _read_nested_mapping(summary, "outcomes")
    latency_outcomes = (
        outcomes.get("latency", {})
        if isinstance(outcomes.get("latency", {}), Mapping)
        else {}
    )
    system = {
        "mean": float(summary.get("mean_latency", latency_outcomes.get("mean", 0.0))),
        "median": float(summary.get("median_latency", latency_outcomes.get("median", 0.0))),
        "p95": float(summary.get("p95_latency", latency_outcomes.get("p95", 0.0))),
        "p99": float(summary.get("p99_latency", latency_outcomes.get("p99", 0.0))),
    }

    breakdown = _read_nested_mapping(summary, "breakdown")
    latency_by_class_raw = summary.get("latency_by_class", breakdown.get("latency_by_class", {}))
    latency_by_worker_raw = summary.get("latency_by_worker", breakdown.get("latency_by_worker", {}))
    if not isinstance(latency_by_class_raw, dict) or not isinstance(latency_by_worker_raw, dict):
        raise ValueError(f"{summary_path} has invalid latency_by_class/latency_by_worker blocks.")

    service_ids = tuple(sorted((str(key) for key in latency_by_class_raw.keys()), key=_class_sort_key))
    worker_ids = tuple(sorted((str(key) for key in latency_by_worker_raw.keys()), key=_class_sort_key))

    can_use_summary_for_entity = True
    for entity in list(latency_by_class_raw.values()) + list(latency_by_worker_raw.values()):
        if not isinstance(entity, dict) or not _summary_entity_has_all_metrics(entity):
            can_use_summary_for_entity = False
            break

    service: Dict[str, Dict[str, float]] = {}
    worker: Dict[str, Dict[str, float]] = {}
    if can_use_summary_for_entity:
        for service_id in service_ids:
            block = _coerce_metric_block(latency_by_class_raw.get(service_id))
            if block is None:
                raise ValueError(f"{summary_path} missing full metric block for service={service_id}")
            service[service_id] = block
        for worker_id in worker_ids:
            block = _coerce_metric_block(latency_by_worker_raw.get(worker_id))
            if block is None:
                raise ValueError(f"{summary_path} missing full metric block for worker={worker_id}")
            worker[worker_id] = block
    else:
        detail_csv = _resolve_detail_csv(run_dir, summary)
        if detail_csv is None:
            raise ValueError(
                f"{run_dir} lacks median/p99 in summary for service/worker metrics and "
                "has no request_detail_metrics.csv for fallback."
            )
        _, by_service, by_worker = _load_detail_stats(detail_csv)
        for service_id in service_ids:
            service[service_id] = by_service.get(service_id, _stats([]))
        for worker_id in worker_ids:
            worker[worker_id] = by_worker.get(worker_id, _stats([]))

    return RunMetrics(
        run_dir=run_dir,
        label=(label_override or _infer_label(run_dir, summary, run_config)),
        system=system,
        service=service,
        worker=worker,
        service_config_canonical=_canonical_json(service_cfg_path),
        worker_config_canonical=_canonical_json(worker_cfg_path),
        service_ids=service_ids,
        worker_ids=worker_ids,
    )


def _parse_run_specs(specs: Sequence[str]) -> List[Tuple[Path, Optional[str]]]:
    parsed: List[Tuple[Path, Optional[str]]] = []
    for spec in specs:
        text = spec.strip()
        if not text:
            raise ValueError("Invalid empty --run value.")
        if "=" in text:
            raw_dir, raw_label = text.split("=", 1)
            label = raw_label.strip()
            if (not raw_dir.strip()) or (not label):
                raise ValueError(
                    f"Invalid --run '{spec}'. When using '=', both folder and label are required."
                )
            for run_dir in _expand_run_dirs(raw_dir):
                parsed.append((run_dir, label))
            continue
        if not text:
            raise ValueError(f"Invalid --run '{spec}'.")
        for run_dir in _expand_run_dirs(text):
            parsed.append((run_dir, None))
    if len(parsed) < 2:
        raise ValueError("At least 2 --run inputs are required for comparison.")
    return parsed


def _dedupe_labels(runs: Sequence[RunMetrics]) -> None:
    seen_labels: Dict[str, int] = {}
    for item in runs:
        label = item.label
        if label in seen_labels:
            seen_labels[label] += 1
            item.label = f"{label}_{seen_labels[label]}"
        else:
            seen_labels[label] = 1


def _validate_compatible_runs(runs: Sequence[RunMetrics]) -> None:
    if not runs:
        raise ValueError("No run metrics loaded.")
    base = runs[0]
    for item in runs[1:]:
        same_service_cfg = item.service_config_canonical == base.service_config_canonical
        same_worker_cfg = item.worker_config_canonical == base.worker_config_canonical
        same_service_ids = item.service_ids == base.service_ids
        same_worker_ids = item.worker_ids == base.worker_ids
        if not (same_service_cfg and same_worker_cfg and same_service_ids and same_worker_ids):
            raise ValueError(
                "Cannot plot comparison because log folders do not share identical "
                "service/worker configuration."
            )


def _plot_grouped_metric_bars(
    ax,
    labels: Sequence[str],
    rows: Sequence[Mapping[str, float]],
    title: str,
) -> None:
    x = list(range(len(METRICS)))
    run_count = max(1, len(labels))
    width = 0.8 / float(run_count)
    max_value = 0.0
    for run_idx, run_label in enumerate(labels):
        values = [float(rows[run_idx].get(metric, 0.0)) for metric in METRICS]
        max_value = max(max_value, max(values, default=0.0))
        offset = (run_idx - (run_count - 1) / 2.0) * width
        bars = ax.bar(
            [value + offset for value in x],
            values,
            width=width,
            label=run_label,
        )
        for bar in bars:
            height = float(bar.get_height())
            y_pad = max(0.005, abs(height) * 0.01)
            ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                height + y_pad,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    if max_value > 0.0:
        ax.set_ylim(0.0, (max_value * 1.20) + 0.01)
    else:
        ax.set_ylim(0.0, 0.1)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_TITLE[metric] for metric in METRICS])
    ax.set_ylabel("Latency (s)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Run / Algorithm")


def plot_system_comparison(runs: Sequence[RunMetrics], out_path: Path, dpi: int) -> None:
    labels = [item.label for item in runs]
    rows = [item.system for item in runs]
    fig, ax = plt.subplots(figsize=(max(9.0, 1.8 * len(runs)), 5.5))
    _plot_grouped_metric_bars(
        ax=ax,
        labels=labels,
        rows=rows,
        title="System Latency Comparison",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_service_comparison(runs: Sequence[RunMetrics], out_path: Path, dpi: int) -> None:
    labels = [item.label for item in runs]
    service_ids = runs[0].service_ids
    fig, axes = plt.subplots(
        len(service_ids),
        1,
        figsize=(max(9.0, 1.8 * len(runs)), max(4.0, 3.4 * len(service_ids))),
        sharex=False,
    )
    if len(service_ids) == 1:
        axes = [axes]
    for ax, service_id in zip(axes, service_ids):
        rows = [item.service.get(service_id, _stats([])) for item in runs]
        _plot_grouped_metric_bars(
            ax=ax,
            labels=labels,
            rows=rows,
            title=f"Service Class {service_id}",
        )
    fig.suptitle("Latency Comparison By Service Class", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_worker_comparison(runs: Sequence[RunMetrics], out_path: Path, dpi: int) -> None:
    labels = [item.label for item in runs]
    worker_ids = runs[0].worker_ids
    fig, axes = plt.subplots(
        len(worker_ids),
        1,
        figsize=(max(9.0, 1.8 * len(runs)), max(4.0, 2.8 * len(worker_ids))),
        sharex=False,
    )
    if len(worker_ids) == 1:
        axes = [axes]
    for ax, worker_id in zip(axes, worker_ids):
        rows = [item.worker.get(worker_id, _stats([])) for item in runs]
        _plot_grouped_metric_bars(
            ax=ax,
            labels=labels,
            rows=rows,
            title=f"Worker {worker_id}",
        )
    fig.suptitle("Latency Comparison By Worker", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare latency metrics across multiple log folders. "
            "Use repeated --run <log_folder> or <log_folder>=<label>."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Input run in format <log_folder> or <log_folder>=<label>. "
            "Supports glob patterns like \"logs/run*\" "
            "(quote pattern to avoid shell expansion). "
            "If label is omitted, tool infers label from policy."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/comparison_plots"),
        help="Output directory for charts (default: logs/comparison_plots).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        specs = _parse_run_specs(args.run)
        runs = [
            _load_run_metrics(run_dir=run_dir, label_override=label)
            for run_dir, label in specs
        ]
        _dedupe_labels(runs)
        _validate_compatible_runs(runs)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    system_out = output_dir / "latency_compare_system.png"
    service_out = output_dir / "latency_compare_by_service.png"
    worker_out = output_dir / "latency_compare_by_worker.png"

    plot_system_comparison(runs=runs, out_path=system_out, dpi=args.dpi)
    plot_service_comparison(runs=runs, out_path=service_out, dpi=args.dpi)
    plot_worker_comparison(runs=runs, out_path=worker_out, dpi=args.dpi)

    print("Generated charts:")
    print(f"- {system_out}")
    print(f"- {service_out}")
    print(f"- {worker_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot basic charts from simulator detail-metrics CSV output."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    raise SystemExit(
        "matplotlib is required for this tool. "
        "Install with: pip install -r tools/requirements.txt"
    ) from error


@dataclass
class DetailMetricSeries:
    """Extracted series from detail metrics CSV."""

    arrivals: List[float] = field(default_factory=list)
    completions: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    arrivals_by_class: Dict[str, List[float]] = field(default_factory=dict)
    completions_by_class: Dict[str, List[float]] = field(default_factory=dict)
    latencies_by_class: Dict[str, List[float]] = field(default_factory=dict)
    latencies_by_worker: Dict[str, List[float]] = field(default_factory=dict)


def _append_by_class(store: Dict[str, List[float]], class_id: str, value: float) -> None:
    bucket = store.setdefault(class_id, [])
    bucket.append(value)


def _load_columns(csv_path: Path) -> DetailMetricSeries:
    series = DetailMetricSeries()

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"t_arrival", "t_done", "latency"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{csv_path} must contain columns: {sorted(required)}"
            )

        for row in reader:
            try:
                t_arrival = float(row["t_arrival"])
                t_done = float(row["t_done"])
                latency = float(row["latency"])
            except (TypeError, ValueError, KeyError):
                continue

            class_id = str(row.get("class_id", "unknown")).strip() or "unknown"
            worker_id = str(row.get("worker_id", "unknown")).strip() or "unknown"
            series.arrivals.append(t_arrival)
            series.completions.append(t_done)
            series.latencies.append(latency)
            _append_by_class(series.arrivals_by_class, class_id, t_arrival)
            _append_by_class(series.completions_by_class, class_id, t_done)
            _append_by_class(series.latencies_by_class, class_id, latency)
            _append_by_class(series.latencies_by_worker, worker_id, latency)

    if not series.arrivals:
        raise ValueError(f"{csv_path} has no valid data rows.")
    return series


def _build_histogram_counts(values: Sequence[float], bin_size: float) -> Tuple[List[float], List[int]]:
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0.")
    max_value = max(values)
    num_bins = max(1, int(max_value // bin_size) + 1)
    edges = [idx * bin_size for idx in range(num_bins + 1)]
    counts = [0 for _ in range(num_bins)]

    for value in values:
        idx = int(value // bin_size)
        idx = min(max(idx, 0), num_bins - 1)
        counts[idx] += 1
    return edges, counts


def _choose_time_scale(max_time_seconds: float) -> Tuple[float, str, str]:
    if max_time_seconds >= 86400.0:
        return 86400.0, "days", "d"
    if max_time_seconds >= 3600.0:
        return 3600.0, "hours", "h"
    if max_time_seconds >= 60.0:
        return 60.0, "minutes", "m"
    return 1.0, "seconds", "s"


def _class_sort_key(raw_class_id: str) -> Tuple[int, object]:
    value = raw_class_id.strip()
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def plot_requests_over_time(
    arrivals: Sequence[float],
    completions: Sequence[float],
    bin_size: float,
    out_path: Path,
    dpi: int,
) -> None:
    arr_edges, arr_counts = _build_histogram_counts(arrivals, bin_size)
    done_edges, done_counts = _build_histogram_counts(completions, bin_size)
    max_time = max(max(arrivals), max(completions))
    scale, axis_unit, _ = _choose_time_scale(max_time)

    arr_x = [value / scale for value in arr_edges[:-1]]
    done_x = [value / scale for value in done_edges[:-1]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.step(arr_x, arr_counts, where="post", label="Arrivals", linewidth=1.8)
    ax.step(done_x, done_counts, where="post", label="Completions", linewidth=1.8)
    ax.set_title(f"Requests Over Time (bin={bin_size:g} s)")
    ax.set_xlabel(f"Simulation time ({axis_unit})")
    ax.set_ylabel("Requests / bin")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_requests_over_time_by_class(
    arrivals_by_class: Dict[str, List[float]],
    bin_size: float,
    out_path: Path,
    dpi: int,
) -> None:
    class_ids = sorted(arrivals_by_class.keys(), key=_class_sort_key)
    if not class_ids:
        return
    all_arrivals = [value for values in arrivals_by_class.values() for value in values]
    max_time = max(all_arrivals) if all_arrivals else 0.0
    scale, axis_unit, _ = _choose_time_scale(max_time)

    fig, axes = plt.subplots(
        len(class_ids),
        1,
        figsize=(11, max(4.0, 3.0 * len(class_ids))),
        sharex=True,
    )
    if len(class_ids) == 1:
        axes = [axes]

    for ax, class_id in zip(axes, class_ids):
        arrivals = arrivals_by_class.get(class_id, [])
        if arrivals:
            arr_edges, arr_counts = _build_histogram_counts(arrivals, bin_size)
            ax.step(
                [value / scale for value in arr_edges[:-1]],
                arr_counts,
                where="post",
                label="Arrivals",
                linewidth=1.6,
            )

        ax.set_title(f"Service class {class_id}")
        ax.set_ylabel("Req/bin")
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel(f"Simulation time ({axis_unit})")
    fig.suptitle(f"Requests Over Time By Service Class (bin={bin_size:g} s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_latency_histogram(
    latencies: Sequence[float],
    out_path: Path,
    bins: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=max(1, bins), edgecolor="black", alpha=0.85)
    ax.set_title("Latency Histogram")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_latency_histogram_by_class(
    latencies_by_class: Dict[str, List[float]],
    out_path: Path,
    bins: int,
    dpi: int,
) -> None:
    class_ids = sorted(latencies_by_class.keys(), key=_class_sort_key)
    if not class_ids:
        return

    fig, axes = plt.subplots(
        len(class_ids),
        1,
        figsize=(8, max(4.0, 3.0 * len(class_ids))),
        sharex=True,
    )
    if len(class_ids) == 1:
        axes = [axes]

    for ax, class_id in zip(axes, class_ids):
        latencies = latencies_by_class.get(class_id, [])
        ax.hist(latencies, bins=max(1, bins), edgecolor="black", alpha=0.85)
        ax.set_title(f"Service class {class_id}")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xlabel("Latency (s)")
    fig.suptitle("Latency Histogram By Service Class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_latency_histogram_by_worker(
    latencies_by_worker: Dict[str, List[float]],
    out_path: Path,
    bins: int,
    dpi: int,
) -> None:
    worker_ids = sorted(latencies_by_worker.keys(), key=_class_sort_key)
    if not worker_ids:
        return

    fig, axes = plt.subplots(
        len(worker_ids),
        1,
        figsize=(8, max(4.0, 2.6 * len(worker_ids))),
        sharex=True,
    )
    if len(worker_ids) == 1:
        axes = [axes]

    for ax, worker_id in zip(axes, worker_ids):
        latencies = latencies_by_worker.get(worker_id, [])
        ax.hist(latencies, bins=max(1, bins), edgecolor="black", alpha=0.85)
        ax.set_title(f"Worker {worker_id}")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xlabel("Latency (s)")
    fig.suptitle("Latency Histogram By Worker")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot charts from request_detail_metrics.csv")
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="Path to request_detail_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for charts (default: <csv_dir>/plots)",
    )
    parser.add_argument(
        "--time-bin",
        type=float,
        default=10.0,
        help="Bin size in seconds for requests-over-time chart",
    )
    parser.add_argument(
        "--latency-bins",
        type=int,
        default=50,
        help="Number of bins for latency histogram",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.detail_csv is None:
        raise SystemExit("Missing input CSV. Use --detail-csv <path>.")
    csv_path = args.detail_csv
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = args.output_dir or (csv_path.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    series = _load_columns(csv_path)

    req_total_plot_path = out_dir / "requests_over_time_total.png"
    req_by_class_plot_path = out_dir / "requests_over_time_by_service_class.png"
    lat_total_plot_path = out_dir / "latency_histogram_total.png"
    lat_by_class_plot_path = out_dir / "latency_histogram_by_service_class.png"
    lat_by_worker_plot_path = out_dir / "latency_histogram_by_worker.png"

    plot_requests_over_time(
        arrivals=series.arrivals,
        completions=series.completions,
        bin_size=args.time_bin,
        out_path=req_total_plot_path,
        dpi=args.dpi,
    )
    plot_requests_over_time_by_class(
        arrivals_by_class=series.arrivals_by_class,
        bin_size=args.time_bin,
        out_path=req_by_class_plot_path,
        dpi=args.dpi,
    )
    plot_latency_histogram(
        latencies=series.latencies,
        out_path=lat_total_plot_path,
        bins=args.latency_bins,
        dpi=args.dpi,
    )
    plot_latency_histogram_by_class(
        latencies_by_class=series.latencies_by_class,
        out_path=lat_by_class_plot_path,
        bins=args.latency_bins,
        dpi=args.dpi,
    )
    plot_latency_histogram_by_worker(
        latencies_by_worker=series.latencies_by_worker,
        out_path=lat_by_worker_plot_path,
        bins=args.latency_bins,
        dpi=args.dpi,
    )

    print(f"Saved: {req_total_plot_path}")
    print(f"Saved: {req_by_class_plot_path}")
    print(f"Saved: {lat_total_plot_path}")
    print(f"Saved: {lat_by_class_plot_path}")
    print(f"Saved: {lat_by_worker_plot_path}")


if __name__ == "__main__":
    main()

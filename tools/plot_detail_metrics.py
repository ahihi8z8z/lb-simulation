#!/usr/bin/env python3
"""Plot basic charts from simulator detail-metrics CSV output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit(
        "matplotlib, numpy, and pandas are required for this tool. "
        "Install with: pip install -r tools/requirements.txt"
    ) from error


@dataclass
class DetailMetricSeries:
    """Extracted series from detail metrics CSV."""

    arrivals: List[float] = field(default_factory=list)
    arrival_job_sizes: List[float] = field(default_factory=list)
    completions: List[float] = field(default_factory=list)
    completion_job_sizes: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    arrivals_by_class: Dict[str, List[float]] = field(default_factory=dict)
    arrival_job_sizes_by_class: Dict[str, List[float]] = field(default_factory=dict)
    completions_by_worker: Dict[str, List[float]] = field(default_factory=dict)
    completion_job_sizes_by_worker: Dict[str, List[float]] = field(default_factory=dict)
    latencies_by_class: Dict[str, List[float]] = field(default_factory=dict)
    latencies_by_worker: Dict[str, List[float]] = field(default_factory=dict)


DETAIL_CSV_FILENAME = "request_detail_metrics.csv"


def _load_columns(csv_path: Path) -> DetailMetricSeries:
    raw_df = pd.read_csv(csv_path)
    required = {"t_arrival", "t_done", "latency", "job_size"}
    if not required.issubset(set(raw_df.columns)):
        raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")

    df = raw_df.copy()
    for column in ("t_arrival", "t_done", "latency", "job_size"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["t_arrival", "t_done", "latency", "job_size"])
    if df.empty:
        raise ValueError(f"{csv_path} has no valid data rows.")

    if "class_id" in df.columns:
        class_keys = (
            df["class_id"]
            .astype(str)
            .str.strip()
            .replace("", "unknown")
            .fillna("unknown")
        )
    else:
        class_keys = pd.Series(["unknown"] * len(df), index=df.index)

    if "worker_id" in df.columns:
        worker_keys = (
            df["worker_id"]
            .astype(str)
            .str.strip()
            .replace("", "unknown")
            .fillna("unknown")
        )
    else:
        worker_keys = pd.Series(["unknown"] * len(df), index=df.index)

    df = df.assign(_class_key=class_keys, _worker_key=worker_keys)
    series = DetailMetricSeries(
        arrivals=df["t_arrival"].tolist(),
        arrival_job_sizes=df["job_size"].tolist(),
        completions=df["t_done"].tolist(),
        completion_job_sizes=df["job_size"].tolist(),
        latencies=df["latency"].tolist(),
        arrivals_by_class={
            str(class_id): group["t_arrival"].tolist()
            for class_id, group in df.groupby("_class_key", sort=False)
        },
        arrival_job_sizes_by_class={
            str(class_id): group["job_size"].tolist()
            for class_id, group in df.groupby("_class_key", sort=False)
        },
        completions_by_worker={
            str(worker_id): group["t_done"].tolist()
            for worker_id, group in df.groupby("_worker_key", sort=False)
        },
        completion_job_sizes_by_worker={
            str(worker_id): group["job_size"].tolist()
            for worker_id, group in df.groupby("_worker_key", sort=False)
        },
        latencies_by_class={
            str(class_id): group["latency"].tolist()
            for class_id, group in df.groupby("_class_key", sort=False)
        },
        latencies_by_worker={
            str(worker_id): group["latency"].tolist()
            for worker_id, group in df.groupby("_worker_key", sort=False)
        },
    )
    return series


def _moving_average_over_time(
    times: Sequence[float],
    values: Sequence[float],
    window_size: float,
) -> Tuple[List[float], List[float]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if len(times) != len(values):
        raise ValueError("times and values must have same length.")
    if not times:
        return [], []

    pairs = sorted(zip(times, values), key=lambda item: item[0])
    sorted_times = [pair[0] for pair in pairs]
    sorted_values = [pair[1] for pair in pairs]
    t_min = sorted_times[0]
    t_max = sorted_times[-1]

    if t_max == t_min:
        return [t_min], [sum(sorted_values) / window_size]

    span = t_max - t_min
    approx_points = int(span / max(window_size / 4.0, 1e-6)) + 1
    num_points = min(1800, max(320, approx_points))
    time_grid = [t_min + span * (idx / (num_points - 1)) for idx in range(num_points)]

    rates: List[float] = []
    left = 0
    right = 0
    running_sum = 0.0
    n = len(sorted_times)

    for current_t in time_grid:
        while right < n and sorted_times[right] <= current_t:
            running_sum += sorted_values[right]
            right += 1

        lower_bound = current_t - window_size
        while left < right and sorted_times[left] < lower_bound:
            running_sum -= sorted_values[left]
            left += 1

        rates.append(running_sum / window_size)

    return time_grid, rates


def _smooth_density(
    centers: np.ndarray,
    densities: np.ndarray,
) -> np.ndarray:
    if centers.size <= 2:
        return densities

    sigma_bins = max(1.0, centers.size / 25.0)
    radius = max(2, int(round(3.0 * sigma_bins)))
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)

    smoothed = np.convolve(densities, kernel, mode="same")
    if smoothed.size != densities.size:
        offset = max(0, (smoothed.size - densities.size) // 2)
        smoothed = smoothed[offset : offset + densities.size]
    integrate = getattr(np, "trapezoid", np.trapz)
    original_area = integrate(densities, centers)
    smooth_area = integrate(smoothed, centers)
    if smooth_area > 0:
        smoothed = smoothed * (original_area / smooth_area)
    return smoothed


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
    arrival_job_sizes: Sequence[float],
    ma_window: float,
    out_path: Path,
    dpi: int,
) -> None:
    req_times, req_rates = _moving_average_over_time(
        arrivals,
        [1.0 for _ in arrivals],
        ma_window,
    )
    job_times, job_rates = _moving_average_over_time(
        arrivals,
        arrival_job_sizes,
        ma_window,
    )
    max_time = max(arrivals) if arrivals else 0.0
    scale, axis_unit, _ = _choose_time_scale(max_time)

    req_x = [value / scale for value in req_times]
    job_x = [value / scale for value in job_times]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax_requests, ax_job_size = axes

    ax_requests.plot(req_x, req_rates, label="Arrivals", linewidth=1.8)
    ax_requests.set_title(f"Requests Over Time (MA window={ma_window:g} s)")
    ax_requests.set_xlabel(f"Simulation time ({axis_unit})")
    ax_requests.set_ylabel("Requests / second")
    ax_requests.grid(True, alpha=0.25)
    ax_requests.legend()

    ax_job_size.plot(job_x, job_rates, label="Arrivals", linewidth=1.8)
    ax_job_size.set_title(f"Job Size Over Time (MA window={ma_window:g} s)")
    ax_job_size.set_xlabel(f"Simulation time ({axis_unit})")
    ax_job_size.set_ylabel("Job size / second")
    ax_job_size.grid(True, alpha=0.25)
    ax_job_size.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_requests_over_time_by_class(
    arrivals_by_class: Dict[str, List[float]],
    arrival_job_sizes_by_class: Dict[str, List[float]],
    ma_window: float,
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
        2,
        figsize=(16, max(4.2, 2.8 * len(class_ids))),
        sharex=True,
    )
    if len(class_ids) == 1:
        axes = [axes.tolist()]
    else:
        axes = axes.tolist()

    for axis_row, class_id in zip(axes, class_ids):
        ax_req, ax_job = axis_row
        arrivals = arrivals_by_class.get(class_id, [])
        job_sizes = arrival_job_sizes_by_class.get(class_id, [])
        req_times, req_rates = _moving_average_over_time(
            arrivals,
            [1.0 for _ in arrivals],
            ma_window,
        )
        job_times, job_rates = _moving_average_over_time(
            arrivals,
            job_sizes,
            ma_window,
        )

        ax_req.plot([value / scale for value in req_times], req_rates, linewidth=1.6)
        ax_req.set_ylabel(f"Class {class_id}\nReq/s")
        ax_req.grid(True, alpha=0.25)

        ax_job.plot([value / scale for value in job_times], job_rates, linewidth=1.6)
        ax_job.set_ylabel("Job size/s")
        ax_job.grid(True, alpha=0.25)

    axes[0][0].set_title("Requests Over Time")
    axes[0][1].set_title("Job Size Over Time")
    axes[-1][0].set_xlabel(f"Simulation time ({axis_unit})")
    axes[-1][1].set_xlabel(f"Simulation time ({axis_unit})")
    fig.suptitle(f"Requests / Job Size Over Time By Service Class (MA window={ma_window:g} s)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_requests_over_time_by_worker(
    completions_by_worker: Dict[str, List[float]],
    completion_job_sizes_by_worker: Dict[str, List[float]],
    ma_window: float,
    out_path: Path,
    dpi: int,
) -> None:
    worker_ids = sorted(completions_by_worker.keys(), key=_class_sort_key)
    if not worker_ids:
        return

    all_times = [value for values in completions_by_worker.values() for value in values]
    max_time = max(all_times) if all_times else 0.0
    scale, axis_unit, _ = _choose_time_scale(max_time)

    fig, axes = plt.subplots(
        len(worker_ids),
        2,
        figsize=(16, max(4.2, 2.6 * len(worker_ids))),
        sharex=True,
    )
    if len(worker_ids) == 1:
        axes = [axes.tolist()]
    else:
        axes = axes.tolist()

    for axis_row, worker_id in zip(axes, worker_ids):
        ax_req, ax_job = axis_row
        completion_times = completions_by_worker.get(worker_id, [])
        completion_sizes = completion_job_sizes_by_worker.get(worker_id, [])
        req_times, req_rates = _moving_average_over_time(
            completion_times,
            [1.0 for _ in completion_times],
            ma_window,
        )
        job_times, job_rates = _moving_average_over_time(
            completion_times,
            completion_sizes,
            ma_window,
        )

        ax_req.plot([value / scale for value in req_times], req_rates, linewidth=1.6)
        ax_req.set_ylabel(f"Worker {worker_id}\nReq/s")
        ax_req.grid(True, alpha=0.25)

        ax_job.plot([value / scale for value in job_times], job_rates, linewidth=1.6)
        ax_job.set_ylabel("Job size/s")
        ax_job.grid(True, alpha=0.25)

    axes[0][0].set_title("Requests Over Completion Time")
    axes[0][1].set_title("Job Size Over Completion Time")
    axes[-1][0].set_xlabel(f"Simulation time ({axis_unit})")
    axes[-1][1].set_xlabel(f"Simulation time ({axis_unit})")
    fig.suptitle(f"Requests / Job Size Over Time By Worker (MA window={ma_window:g} s)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_latency_hist_with_smooth_pdf(
    ax: "plt.Axes",
    latencies: Sequence[float],
) -> None:
    densities, edges, _ = ax.hist(
        latencies,
        bins="auto",
        density=True,
        edgecolor="black",
        alpha=0.35,
        label="Histogram density",
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    smoothed = _smooth_density(centers, densities)
    ax.plot(centers, smoothed, linewidth=1.8, label="Smoothed PDF")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()


def plot_latency_histogram(
    latencies: Sequence[float],
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_latency_hist_with_smooth_pdf(ax, latencies)
    ax.set_title("Latency Histogram")
    ax.set_xlabel("Latency (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_latency_histogram_by_class(
    latencies_by_class: Dict[str, List[float]],
    out_path: Path,
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
        _plot_latency_hist_with_smooth_pdf(ax, latencies)
        ax.set_title(f"Service class {class_id}")

    axes[-1].set_xlabel("Latency (s)")
    fig.suptitle("Latency Histogram By Service Class")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_latency_histogram_by_worker(
    latencies_by_worker: Dict[str, List[float]],
    out_path: Path,
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
        _plot_latency_hist_with_smooth_pdf(ax, latencies)
        ax.set_title(f"Worker {worker_id}")

    axes[-1].set_xlabel("Latency (s)")
    fig.suptitle("Latency Histogram By Worker")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot charts from request_detail_metrics.csv")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help=(
            "Path to one request_detail_metrics.csv "
            "(or a folder containing many such CSV files)"
        ),
    )
    input_group.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Root folder to recursively scan for request_detail_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for charts. Single CSV: write directly here. "
            "Folder/batch mode: write to <output_dir>/<relative_run_dir>/plots. "
            "Default: <csv_dir>/plots."
        ),
    )
    parser.add_argument(
        "--ma-window",
        type=float,
        default=10.0,
        help="Moving-average window in seconds for over-time charts",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI")
    return parser


def _find_detail_csvs(input_path: Path) -> Tuple[List[Path], Optional[Path]]:
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")
    if input_path.is_file():
        return [input_path], None
    csv_paths = sorted(input_path.rglob(DETAIL_CSV_FILENAME))
    return csv_paths, input_path


def _build_output_dir(
    csv_path: Path,
    output_dir: Optional[Path],
    batch_root: Optional[Path],
) -> Path:
    if output_dir is None:
        return csv_path.parent / "plots"
    if batch_root is None:
        return output_dir
    relative_run_dir = csv_path.parent.relative_to(batch_root)
    return output_dir / relative_run_dir / "plots"


def _plot_one_csv(
    csv_path: Path,
    out_dir: Path,
    ma_window: float,
    dpi: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    series = _load_columns(csv_path)

    req_total_plot_path = out_dir / "requests_over_time_total.png"
    req_by_class_plot_path = out_dir / "requests_over_time_by_service_class.png"
    req_by_worker_plot_path = out_dir / "requests_job_size_over_time_by_worker.png"
    lat_total_plot_path = out_dir / "latency_histogram_total.png"
    lat_by_class_plot_path = out_dir / "latency_histogram_by_service_class.png"
    lat_by_worker_plot_path = out_dir / "latency_histogram_by_worker.png"

    output_paths = [
        req_total_plot_path,
        req_by_class_plot_path,
        req_by_worker_plot_path,
        lat_total_plot_path,
        lat_by_class_plot_path,
        lat_by_worker_plot_path,
    ]

    plot_requests_over_time(
        arrivals=series.arrivals,
        arrival_job_sizes=series.arrival_job_sizes,
        ma_window=ma_window,
        out_path=req_total_plot_path,
        dpi=dpi,
    )
    plot_requests_over_time_by_class(
        arrivals_by_class=series.arrivals_by_class,
        arrival_job_sizes_by_class=series.arrival_job_sizes_by_class,
        ma_window=ma_window,
        out_path=req_by_class_plot_path,
        dpi=dpi,
    )
    plot_requests_over_time_by_worker(
        completions_by_worker=series.completions_by_worker,
        completion_job_sizes_by_worker=series.completion_job_sizes_by_worker,
        ma_window=ma_window,
        out_path=req_by_worker_plot_path,
        dpi=dpi,
    )
    plot_latency_histogram(
        latencies=series.latencies,
        out_path=lat_total_plot_path,
        dpi=dpi,
    )
    plot_latency_histogram_by_class(
        latencies_by_class=series.latencies_by_class,
        out_path=lat_by_class_plot_path,
        dpi=dpi,
    )
    plot_latency_histogram_by_worker(
        latencies_by_worker=series.latencies_by_worker,
        out_path=lat_by_worker_plot_path,
        dpi=dpi,
    )
    return output_paths


def main() -> None:
    args = build_parser().parse_args()

    input_path = args.logs_dir if args.logs_dir is not None else args.detail_csv
    if input_path is None:
        raise SystemExit("Missing input. Use --detail-csv <file|folder> or --logs-dir <folder>.")

    csv_paths, batch_root = _find_detail_csvs(input_path)
    if not csv_paths:
        raise SystemExit(f"No {DETAIL_CSV_FILENAME} found under: {input_path}")

    total = len(csv_paths)
    for index, csv_path in enumerate(csv_paths, start=1):
        out_dir = _build_output_dir(
            csv_path=csv_path,
            output_dir=args.output_dir,
            batch_root=batch_root,
        )
        saved_paths = _plot_one_csv(
            csv_path=csv_path,
            out_dir=out_dir,
            ma_window=args.ma_window,
            dpi=args.dpi,
        )
        if total > 1:
            print(f"[{index}/{total}] Source: {csv_path}")
        for saved_path in saved_paths:
            print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()

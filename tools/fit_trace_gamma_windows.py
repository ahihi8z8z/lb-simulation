#!/usr/bin/env python3
"""Fit gamma windows + global Zipf params from a trace CSV."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DEFAULT_GAMMA_WINDOW = "20m"
ZIPF_S_MIN = 1.01
ZIPF_S_MAX = 4.00
ZIPF_COARSE_STEP = 0.02
ZIPF_FINE_STEP = 0.002
_DURATION_PATTERN = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]*)\s*$")


def _normalize_column_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _resolve_timestamp_column(columns: List[str]) -> str:
    normalized_to_raw = {_normalize_column_name(col): col for col in columns if col}
    ts_col = normalized_to_raw.get("timestamp")
    if ts_col is None:
        raise ValueError("Input CSV is missing required 'Timestamp' column.")
    return ts_col


def _resolve_request_tokens_column(columns: List[str]) -> str:
    normalized_to_raw = {_normalize_column_name(col): col for col in columns if col}
    token_col = normalized_to_raw.get("request tokens")
    if token_col is None:
        raise ValueError("Input CSV is missing required 'Request tokens' column.")
    return token_col


def _fit_gamma_moments(samples: List[float]) -> tuple[float, float]:
    # Gamma(shape=alpha, scale=beta): mean=alpha*beta, var=alpha*beta^2
    if len(samples) < 2:
        return float("nan"), float("nan")
    mean = sum(samples) / float(len(samples))
    if mean <= 0:
        return float("nan"), float("nan")
    var = sum((value - mean) ** 2 for value in samples) / float(len(samples))
    if var <= 0:
        return float("nan"), float("nan")
    alpha = (mean * mean) / var
    beta = var / mean
    return alpha, beta


def _parse_duration_seconds(raw: str, arg_name: str) -> float:
    text = str(raw).strip().lower()
    match = _DURATION_PATTERN.match(text)
    if not match:
        raise ValueError(f"Invalid {arg_name} value: {raw}")

    value = float(match.group("value"))
    unit = match.group("unit")
    if unit in {"", "s", "sec", "second", "seconds", "giay"}:
        multiplier = 1.0
    elif unit in {"m", "min", "minute", "minutes", "phut"}:
        multiplier = 60.0
    elif unit in {"h", "hr", "hour", "hours", "gio"}:
        multiplier = 3600.0
    elif unit in {"d", "day", "days", "ngay"}:
        multiplier = 86400.0
    elif unit in {"mo", "mon", "month", "months", "thang"}:
        multiplier = 30.0 * 86400.0
    else:
        raise ValueError(
            f"Unsupported unit in {arg_name}: {raw}. "
            "Use s/m/h/d/mo (or sec/min/hour/day/month variants)."
        )

    seconds = value * multiplier
    if seconds <= 0:
        raise ValueError(f"{arg_name} must be > 0.")
    return seconds


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = float(start)
    while value <= (stop + 1e-12):
        yield value
        value += step


def _zipf_log_likelihood(
    s: float,
    *,
    log_values: List[float],
    xmin: int,
    xmax: int,
    sample_count: int,
    sum_log_x: float,
) -> float:
    del xmin, xmax  # already folded into log_values
    harmonic = sum(math.exp(-s * log_k) for log_k in log_values)
    if harmonic <= 0.0:
        return float("-inf")
    return (-s * sum_log_x) - (sample_count * math.log(harmonic))


def _fit_zipf_s_truncated(request_tokens: pd.Series) -> Dict[str, float]:
    tokens = request_tokens[request_tokens > 0]
    if tokens.empty:
        return {
            "sample_count": 0.0,
            "xmin": float("nan"),
            "max": float("nan"),
            "s": float("nan"),
        }

    counts = tokens.value_counts(sort=False)
    xmin = int(counts.index.min())
    xmax = int(counts.index.max())
    sample_count = int(counts.sum())
    sum_log_x = float((counts.index.to_series().map(math.log) * counts).sum())

    if xmin <= 0 or xmax <= xmin or sample_count <= 0:
        return {
            "sample_count": float(sample_count),
            "xmin": float(xmin),
            "max": float(xmax),
            "s": float("nan"),
        }

    log_values = [math.log(k) for k in range(xmin, xmax + 1)]
    best_s = float("nan")
    best_ll = float("-inf")
    for s in _frange(ZIPF_S_MIN, ZIPF_S_MAX, ZIPF_COARSE_STEP):
        ll = _zipf_log_likelihood(
            s,
            log_values=log_values,
            xmin=xmin,
            xmax=xmax,
            sample_count=sample_count,
            sum_log_x=sum_log_x,
        )
        if ll > best_ll:
            best_ll = ll
            best_s = s

    if math.isnan(best_s):
        return {
            "sample_count": float(sample_count),
            "xmin": float(xmin),
            "max": float(xmax),
            "s": float("nan"),
        }

    fine_min = max(ZIPF_S_MIN, best_s - 2.0 * ZIPF_COARSE_STEP)
    fine_max = min(ZIPF_S_MAX, best_s + 2.0 * ZIPF_COARSE_STEP)
    for s in _frange(fine_min, fine_max, ZIPF_FINE_STEP):
        ll = _zipf_log_likelihood(
            s,
            log_values=log_values,
            xmin=xmin,
            xmax=xmax,
            sample_count=sample_count,
            sum_log_x=sum_log_x,
        )
        if ll > best_ll:
            best_ll = ll
            best_s = s

    return {
        "sample_count": float(sample_count),
        "xmin": float(xmin),
        "max": float(xmax),
        "s": float(best_s),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split trace into configurable windows, fit gamma(alpha,beta) from "
            "inter-arrival times per window, and also fit Zipf from request "
            "tokens on full trace."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input trace CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV file for gamma windows "
            "(default: <input_dir>/<input_stem>_gamma_windows.csv)."
        ),
    )
    parser.add_argument(
        "--zipf-output",
        type=Path,
        default=None,
        help=(
            "Output TXT file for global Zipf params "
            "(default: <input_dir>/<input_stem>_zipf.txt)."
        ),
    )
    parser.add_argument(
        "--gamma-window",
        type=str,
        default=DEFAULT_GAMMA_WINDOW,
        help=(
            "Window size for gamma fitting, supports units "
            "(e.g. 20m, 10m, 1h, 900s). Default: 20m."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = args.input
    gamma_output_path = (
        args.output
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_gamma_windows.csv")
    )
    zipf_output_path = (
        args.zipf_output
        if args.zipf_output is not None
        else input_path.with_name(f"{input_path.stem}_zipf.txt")
    )

    if not input_path.exists() or not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    try:
        window_seconds = _parse_duration_seconds(args.gamma_window, "--gamma-window")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    try:
        header_df = pd.read_csv(input_path, nrows=0)
    except Exception as error:
        raise SystemExit(f"Failed to read CSV header: {error}") from error

    columns = [str(col) for col in header_df.columns]
    if not columns:
        raise SystemExit(f"Input CSV has no header: {input_path}")

    try:
        ts_col = _resolve_timestamp_column(columns)
        request_tokens_col = _resolve_request_tokens_column(columns)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    df = pd.read_csv(
        input_path,
        usecols=[ts_col, request_tokens_col],
        dtype=str,
        keep_default_na=False,
    )

    ts_series = pd.to_numeric(df[ts_col].astype(str).str.strip(), errors="coerce")
    ts_series = ts_series[ts_series.notna()]
    if ts_series.empty:
        raise SystemExit("No valid timestamp rows found in input CSV.")

    ts_series = ts_series.astype(float)
    ts_series = ts_series[ts_series >= 0.0]
    if ts_series.empty:
        raise SystemExit("No non-negative timestamps found in input CSV.")

    request_tokens_series = pd.to_numeric(
        df[request_tokens_col].astype(str).str.strip().str.replace(",", "", regex=False),
        errors="coerce",
    )
    request_tokens_series = request_tokens_series[request_tokens_series.notna()].astype(float)
    request_tokens_series = request_tokens_series[request_tokens_series > 0.0]
    if request_tokens_series.empty:
        raise SystemExit("No valid positive request tokens found in input CSV.")
    request_tokens_series = request_tokens_series.round().astype(int)

    timestamps = sorted(ts_series.tolist())
    base_ts = timestamps[0]
    last_ts = timestamps[-1]
    window_count = int(math.floor((last_ts - base_ts) / window_seconds)) + 1

    per_window_timestamps: Dict[int, List[float]] = {idx: [] for idx in range(window_count)}
    for timestamp in timestamps:
        window_idx = int((timestamp - base_ts) // window_seconds)
        per_window_timestamps[window_idx].append(timestamp)

    rows: List[dict] = []
    for window_idx in range(window_count):
        window_start = base_ts + window_idx * window_seconds
        window_end = window_start + window_seconds
        window_timestamps = per_window_timestamps.get(window_idx, [])

        inter_arrivals = [
            window_timestamps[idx] - window_timestamps[idx - 1]
            for idx in range(1, len(window_timestamps))
        ]
        positive_inter_arrivals = [value for value in inter_arrivals if value > 0]
        alpha, beta = _fit_gamma_moments(positive_inter_arrivals)

        rows.append(
            {
                "window_index": window_idx,
                "window_seconds": window_seconds,
                "window_start_timestamp": window_start,
                "window_end_timestamp": window_end,
                "request_count": len(window_timestamps),
                "inter_arrival_count": len(positive_inter_arrivals),
                "gamma": alpha,
                "alpha": alpha,
                "beta": beta,
            }
        )

    zipf_fit = _fit_zipf_s_truncated(request_tokens_series)

    gamma_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(gamma_output_path, index=False)

    zipf_output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipf_output_path.open("w", encoding="utf-8") as file:
        file.write(f"input_csv={input_path}\n")
        file.write(f"request_tokens_column={request_tokens_col}\n")
        file.write(f"sample_count={int(zipf_fit['sample_count'])}\n")
        if math.isnan(zipf_fit["s"]):
            file.write("fit_status=failed\n")
            file.write("reason=insufficient_range_or_invalid_tokens\n")
        else:
            file.write("fit_status=ok\n")
            file.write("distribution=truncated_discrete_zipf\n")
            file.write(f"s={zipf_fit['s']:.6f}\n")
            file.write(f"xmin={int(zipf_fit['xmin'])}\n")
            file.write(f"max={int(zipf_fit['max'])}\n")
            file.write(f"search_s_min={ZIPF_S_MIN}\n")
            file.write(f"search_s_max={ZIPF_S_MAX}\n")
            file.write(f"coarse_step={ZIPF_COARSE_STEP}\n")
            file.write(f"fine_step={ZIPF_FINE_STEP}\n")

    print(f"Saved {len(rows)} gamma windows to {gamma_output_path}")
    print(f"Saved Zipf params to {zipf_output_path}")


if __name__ == "__main__":
    main()

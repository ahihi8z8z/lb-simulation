#!/usr/bin/env python3
"""Extract a time window from a trace CSV and save it as a new CSV file."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _normalize_column_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def _resolve_timestamp_column(fieldnames: List[str]) -> str:
    normalized_to_raw: Dict[str, str] = {
        _normalize_column_name(field): field for field in fieldnames if field
    }
    ts_col = normalized_to_raw.get("timestamp")
    if ts_col is None:
        raise ValueError("Input CSV is missing required 'Timestamp' column.")
    return ts_col


_DURATION_PATTERN = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]*)\s*$")


def _parse_duration_seconds(raw: str, arg_name: str) -> float:
    """Parse duration text into seconds.

    Supported units:
    - seconds: s, sec, second, seconds, giay
    - minutes: m, min, minute, minutes, phut
    - hours: h, hr, hour, hours, gio
    - days: d, day, days, ngay
    - months (30 days): mo, mon, month, months, thang
    """

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
    if seconds < 0:
        raise ValueError(f"{arg_name} must be >= 0.")
    return seconds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract rows in a time window from trace CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input trace CSV path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--window",
        required=True,
        type=str,
        help=(
            "Window length (supports units), e.g. 1h, 2d, 1mo, 1800, 30m."
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        type=str,
        help=(
            "Window start time (supports units). If omitted, start is sampled randomly."
        ),
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Random seed used when --start is omitted.",
    )
    parser.add_argument(
        "--exclude-end",
        action="store_true",
        help="If set, keep rows with start <= Timestamp < end.",
    )
    return parser


def _in_window(value: float, start: float, end: float, exclude_end: bool) -> bool:
    if exclude_end:
        return (start <= value) and (value < end)
    return (start <= value) and (value <= end)


def _scan_timestamp_bounds(input_path: Path) -> Tuple[List[str], str, float, float]:
    min_ts: Optional[float] = None
    max_ts: Optional[float] = None

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise SystemExit(f"Input CSV has no header: {input_path}")
        fieldnames = list(reader.fieldnames)
        ts_col = _resolve_timestamp_column(fieldnames)

        for row in reader:
            ts_raw = (row.get(ts_col) or "").strip()
            try:
                ts = float(ts_raw)
            except ValueError:
                continue
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts

    if min_ts is None or max_ts is None:
        raise SystemExit("No valid Timestamp rows found in input CSV.")
    return fieldnames, ts_col, min_ts, max_ts


def main() -> None:
    args = _build_parser().parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    try:
        window_seconds = _parse_duration_seconds(args.window, "--window")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if window_seconds <= 0:
        raise SystemExit("--window must be > 0.")

    start_seconds: Optional[float] = None
    if args.start is not None:
        try:
            start_seconds = _parse_duration_seconds(args.start, "--start")
        except ValueError as error:
            raise SystemExit(str(error)) from error

    fieldnames, ts_col, min_ts, max_ts = _scan_timestamp_bounds(args.input)
    latest_start = max_ts - window_seconds
    if latest_start < min_ts:
        raise SystemExit(
            f"Window too long ({window_seconds}s). "
            f"Trace span is only [{min_ts}, {max_ts}] ({max_ts - min_ts}s)."
        )

    sampled_randomly = start_seconds is None
    if start_seconds is None:
        rng = random.Random(args.seed)
        start_seconds = rng.uniform(min_ts, latest_start)
    assert start_seconds is not None
    if start_seconds < min_ts or start_seconds > latest_start:
        raise SystemExit(
            f"--start={start_seconds} is out of valid range [{min_ts}, {latest_start}] "
            f"for window length {window_seconds}s."
        )
    end_seconds = start_seconds + window_seconds

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    skipped_invalid_ts = 0

    with args.input.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise SystemExit(f"Input CSV has no header: {args.input}")

        with args.output.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                ts_raw = (row.get(ts_col) or "").strip()
                try:
                    timestamp = float(ts_raw)
                except ValueError:
                    skipped_invalid_ts += 1
                    continue
                if _in_window(timestamp, start_seconds, end_seconds, args.exclude_end):
                    writer.writerow(row)
                    kept_rows += 1

    bracket_right = ")" if args.exclude_end else "]"
    sampling_mode = "random" if sampled_randomly else "fixed_start"
    print(
        f"Saved {kept_rows} rows to {args.output} "
        f"for window [{start_seconds}, {end_seconds}{bracket_right} "
        f"(mode={sampling_mode})"
    )
    print(
        f"Scanned {total_rows} rows, skipped invalid timestamp rows: {skipped_invalid_ts}"
    )


if __name__ == "__main__":
    main()

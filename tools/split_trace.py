#!/usr/bin/env python3
"""
Split a trace CSV into multiple files by categorical columns (e.g., Log Type, Model).

Usage:
  python tools/split_trace.py \
    --input traces/BurstGPT_without_fails_1.csv \
    --columns "Log Type" "Model" \
    --output-dir traces/splits

The script streams the input file and writes one output CSV per unique combination
of the specified columns. Filenames are slugified to stay filesystem-safe.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split trace CSV into multiple files by given columns."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=pathlib.Path,
        help="Path to input trace CSV.",
    )
    parser.add_argument(
        "--columns",
        required=True,
        nargs="+",
        help="Column names to split by (e.g., 'Log Type' 'Model').",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory to write split CSV files.",
    )
    parser.add_argument(
        "--unknown-value",
        default="UNKNOWN",
        help="Value to use when a row is missing a required column.",
    )
    return parser.parse_args()


_SAFE_CHARS = re.compile(r"[^a-z0-9_-]+")


def _slugify(value: str) -> str:
    lowered = value.strip().lower().replace(" ", "_")
    cleaned = _SAFE_CHARS.sub("-", lowered)
    return cleaned or "empty"


def _output_name(
    input_path: pathlib.Path, columns: Sequence[str], values: Sequence[str]
) -> str:
    parts = []
    for col, val in zip(columns, values):
        parts.append(f"{_slugify(col)}-{_slugify(val)}")
    base = input_path.stem
    suffix = "__".join(parts)
    return f"{base}__{suffix}.csv"


def split_trace(
    input_path: pathlib.Path,
    columns: Sequence[str],
    output_dir: pathlib.Path,
    unknown_value: str,
) -> Dict[Tuple[str, ...], int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        header = reader.fieldnames
        if header is None:
            raise ValueError("Input CSV has no header.")

        missing_cols = [col for col in columns if col not in header]
        if missing_cols:
            raise ValueError(f"Columns not found in CSV header: {missing_cols}")

        writers: Dict[Tuple[str, ...], csv.DictWriter] = {}
        handles: Dict[Tuple[str, ...], object] = {}
        counts: Dict[Tuple[str, ...], int] = defaultdict(int)

        for row in reader:
            key_values = tuple(row.get(col, unknown_value) or unknown_value for col in columns)
            if key_values not in writers:
                out_name = _output_name(input_path, columns, key_values)
                out_path = output_dir / out_name
                handle = out_path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writers[key_values] = writer
                handles[key_values] = handle
            writers[key_values].writerow(row)
            counts[key_values] += 1

        for handle in handles.values():
            handle.close()
    return counts


def main() -> None:
    args = parse_args()
    counts = split_trace(
        input_path=args.input,
        columns=args.columns,
        output_dir=args.output_dir,
        unknown_value=args.unknown_value,
    )
    print(f"Wrote {len(counts)} split files to {args.output_dir}")
    for key, count in sorted(counts.items()):
        label = ", ".join(f"{col}={val}" for col, val in zip(args.columns, key))
        print(f"  {label}: {count} rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract LP-related worker-weight updates from runtime.log into a CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple


_LOG_RECORD_RE = re.compile(
    r"^(?P<timestamp>t=\s*(?P<seconds>[^\s]+)s)\s+\|\s+"
    r"(?P<level>[A-Z]+)\s+\|\s+(?P<logger>[^|]+?)\s+\|\s+(?P<message>.*)$"
)
_LB_ID_RE = re.compile(r"\blb_id=(?P<lb_id>[A-Za-z0-9_.-]+)")
_CLASS_ID_RE = re.compile(r"\bclass_id=(?P<class_id>\d+)\b")
_CLASS_LB_ID_RE = re.compile(r"^class_(?P<class_id>\d+)$")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_UPDATED_WORKER_WEIGHTS_PREFIX = "Updated worker weights for main load balancer"
_KEYWORDS: Tuple[str, ...] = ("lp_weights", "wrr_lp_weight_matrix")
_BRACKET_PAIRS = {"[": "]", "{": "}"}


@dataclass
class RuntimeLogRecord:
    """One runtime.log record, with multi-line messages already reconstructed."""

    record_index: int
    timestamp: str
    timestamp_seconds: Optional[float]
    level: str
    logger_name: str
    message: str


@dataclass
class ExtractedWeightRow:
    """One output CSV row representing a single weight vector."""

    record_index: int
    timestamp: str
    timestamp_seconds: Optional[float]
    level: str
    logger_name: str
    event_type: str
    lb_id: str
    class_id: str
    row_index: Optional[int]
    weights: List[float]
    source_message: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a simulator runtime.log, collect LP-related worker-weight values by "
            "timestamp, and export them to CSV."
        )
    )
    parser.add_argument(
        "runtime_log",
        type=Path,
        help="Path to the input runtime.log file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. If omitted, defaults to "
            "<runtime_log_dir>/<runtime_log_stem>_lp_weights.csv."
        ),
    )
    return parser


def _default_output_path(runtime_log_path: Path) -> Path:
    return runtime_log_path.with_name(f"{runtime_log_path.stem}_lp_weights.csv")


def _parse_optional_float(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except ValueError:
        return None


def _load_runtime_records(runtime_log_path: Path) -> List[RuntimeLogRecord]:
    records: List[RuntimeLogRecord] = []
    current: Optional[RuntimeLogRecord] = None

    with runtime_log_path.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")
            match = _LOG_RECORD_RE.match(line)
            if match:
                if current is not None:
                    records.append(current)
                current = RuntimeLogRecord(
                    record_index=len(records),
                    timestamp=match.group("timestamp"),
                    timestamp_seconds=_parse_optional_float(match.group("seconds")),
                    level=match.group("level"),
                    logger_name=match.group("logger").strip(),
                    message=match.group("message"),
                )
                continue

            if current is None:
                continue
            current.message = f"{current.message}\n{line}"

    if current is not None:
        records.append(current)
    return records


def _extract_balanced_segment(text: str, start_index: int) -> Optional[str]:
    if start_index < 0 or start_index >= len(text):
        return None
    open_char = text[start_index]
    if open_char not in _BRACKET_PAIRS:
        return None

    stack: List[str] = []
    in_quote: Optional[str] = None
    escaped = False

    for index in range(start_index, len(text)):
        char = text[index]
        if in_quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_quote:
                in_quote = None
            continue

        if char in {"'", '"'}:
            in_quote = char
            continue

        if char in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[char])
            continue

        if stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return text[start_index : index + 1]
            continue

        if char in _BRACKET_PAIRS.values():
            return None

    return None


def _iter_payloads_after(text: str, start_index: int):
    index = start_index
    while index < len(text):
        if text[index] not in _BRACKET_PAIRS:
            index += 1
            continue
        payload = _extract_balanced_segment(text, index)
        if payload is None:
            index += 1
            continue
        yield payload
        index += len(payload)


def _find_payload_after(text: str, start_index: int) -> Optional[str]:
    return next(_iter_payloads_after(text, start_index), None)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_sequence(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(_is_number(item) for item in value)
    )


def _parse_numeric_array_text(payload: str) -> object:
    text = payload.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("Numeric array payload must start with '[' and end with ']'.")

    inner = text[1:-1].strip()
    if not inner:
        return []

    nested_segments: List[str] = []
    other_chars: List[str] = []
    depth = 0
    segment_start: Optional[int] = None
    for index, char in enumerate(inner):
        if char == "[":
            if depth == 0:
                segment_start = index
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("Malformed bracket array payload.")
            if depth == 0 and segment_start is not None:
                nested_segments.append(inner[segment_start : index + 1])
                segment_start = None
            continue
        if depth == 0:
            other_chars.append(char)

    if depth != 0:
        raise ValueError("Malformed bracket array payload.")

    other_text = "".join(other_chars)
    if nested_segments and not _NUMBER_RE.search(other_text):
        return [_parse_numeric_array_text(segment) for segment in nested_segments]

    numbers = [float(match.group(0)) for match in _NUMBER_RE.finditer(inner)]
    if numbers:
        return numbers

    raise ValueError("Unable to parse numeric array payload.")


def _parse_payload(payload: str) -> object:
    stripped = payload.strip()
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        if stripped.startswith("[") and stripped.endswith("]"):
            return _parse_numeric_array_text(stripped)
        raise


def _coerce_weight_rows(
    value: object,
) -> List[Tuple[Optional[str], Optional[int], List[float]]]:
    if _is_numeric_sequence(value):
        return [(None, None, [float(item) for item in value])]

    if isinstance(value, Mapping):
        rows: List[Tuple[Optional[str], Optional[int], List[float]]] = []
        for key, inner_value in value.items():
            inner_rows = _coerce_weight_rows(inner_value)
            for class_override, row_index, weights in inner_rows:
                class_id = class_override if class_override is not None else str(key)
                rows.append((class_id, row_index, weights))
        return rows

    if isinstance(value, (list, tuple)):
        rows = []
        for row_index, item in enumerate(value):
            if not _is_numeric_sequence(item):
                return []
            rows.append((None, row_index, [float(inner) for inner in item]))
        return rows

    return []


def _extract_lb_id(message: str) -> str:
    match = _LB_ID_RE.search(message)
    if match is None:
        return ""
    return match.group("lb_id")


def _extract_class_id(message: str, lb_id: str) -> str:
    match = _CLASS_ID_RE.search(message)
    if match is not None:
        return match.group("class_id")

    lb_match = _CLASS_LB_ID_RE.match(lb_id)
    if lb_match is not None:
        return lb_match.group("class_id")
    return ""


def _build_rows_from_payload(
    record: RuntimeLogRecord,
    payload: str,
    event_type: str,
) -> List[ExtractedWeightRow]:
    try:
        parsed_payload = _parse_payload(payload)
    except (SyntaxError, ValueError):
        return []

    row_specs = _coerce_weight_rows(parsed_payload)
    if not row_specs:
        return []

    lb_id = _extract_lb_id(record.message)
    base_class_id = _extract_class_id(record.message, lb_id)
    source_message = record.message.replace("\n", "\\n")

    rows: List[ExtractedWeightRow] = []
    for class_override, row_index, weights in row_specs:
        class_id = class_override if class_override is not None else base_class_id
        rows.append(
            ExtractedWeightRow(
                record_index=record.record_index,
                timestamp=record.timestamp,
                timestamp_seconds=record.timestamp_seconds,
                level=record.level,
                logger_name=record.logger_name,
                event_type=event_type,
                lb_id=lb_id,
                class_id=class_id,
                row_index=row_index,
                weights=weights,
                source_message=source_message,
            )
        )
    return rows


def _extract_from_updated_worker_weights(
    record: RuntimeLogRecord,
) -> List[ExtractedWeightRow]:
    if _UPDATED_WORKER_WEIGHTS_PREFIX not in record.message:
        return []
    payload = _find_payload_after(
        record.message,
        record.message.find(_UPDATED_WORKER_WEIGHTS_PREFIX) + len(_UPDATED_WORKER_WEIGHTS_PREFIX),
    )
    if payload is None:
        return []
    return _build_rows_from_payload(
        record=record,
        payload=payload,
        event_type="worker_weights_update",
    )


def _extract_from_keyword(
    record: RuntimeLogRecord,
    keyword: str,
) -> List[ExtractedWeightRow]:
    lowered_message = record.message.lower()
    lowered_keyword = keyword.lower()
    search_start = 0
    while True:
        keyword_index = lowered_message.find(lowered_keyword, search_start)
        if keyword_index < 0:
            return []
        for payload in _iter_payloads_after(record.message, keyword_index + len(keyword)):
            rows = _build_rows_from_payload(record=record, payload=payload, event_type=keyword)
            if rows:
                return rows
        search_start = keyword_index + len(keyword)


def _extract_weight_rows(records: Sequence[RuntimeLogRecord]) -> List[ExtractedWeightRow]:
    rows: List[ExtractedWeightRow] = []
    for record in records:
        extracted = _extract_from_updated_worker_weights(record)
        if extracted:
            rows.extend(extracted)
            continue

        for keyword in _KEYWORDS:
            extracted = _extract_from_keyword(record, keyword)
            if extracted:
                rows.extend(extracted)
                break
    return rows


def _write_csv(output_path: Path, rows: Sequence[ExtractedWeightRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_weights = max((len(row.weights) for row in rows), default=0)
    fieldnames = [
        "record_index",
        "timestamp",
        "timestamp_seconds",
        "level",
        "logger_name",
        "event_type",
        "lb_id",
        "class_id",
        "row_index",
        "weight_count",
    ]
    fieldnames.extend(f"worker_{index}" for index in range(max_weights))
    fieldnames.append("source_message")

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {
                "record_index": row.record_index,
                "timestamp": row.timestamp,
                "timestamp_seconds": (
                    "" if row.timestamp_seconds is None else f"{row.timestamp_seconds:.6f}"
                ),
                "level": row.level,
                "logger_name": row.logger_name,
                "event_type": row.event_type,
                "lb_id": row.lb_id,
                "class_id": row.class_id,
                "row_index": "" if row.row_index is None else row.row_index,
                "weight_count": len(row.weights),
                "source_message": row.source_message,
            }
            for index, weight in enumerate(row.weights):
                csv_row[f"worker_{index}"] = f"{weight:.12g}"
            writer.writerow(csv_row)


def main() -> int:
    args = _build_parser().parse_args()
    runtime_log_path = args.runtime_log.expanduser().resolve()
    if not runtime_log_path.exists() or not runtime_log_path.is_file():
        raise SystemExit(f"runtime.log file does not exist: {runtime_log_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(runtime_log_path)
    )
    records = _load_runtime_records(runtime_log_path)
    rows = _extract_weight_rows(records)
    _write_csv(output_path, rows)
    print(f"Extracted {len(rows)} lp-weight rows to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared logging setup for simulator modules."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Dict, Optional

_LEVEL_MAP: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_SIM_TIME_PROVIDER: Optional[Callable[[], float]] = None


class SimulationTimeFormatter(logging.Formatter):
    """Formatter that renders record time from simulation clock instead of wallclock."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        del record, datefmt
        provider = _SIM_TIME_PROVIDER
        if provider is None:
            return "t=     0.000s"
        try:
            value = float(provider())
        except Exception:
            return "t=         ?s"
        if not math.isfinite(value):
            return "t=         ?s"
        return f"t={value:10.3f}s"


def set_simulation_time_provider(provider: Optional[Callable[[], float]]) -> None:
    """Set a callback that returns current simulation time in seconds."""

    global _SIM_TIME_PROVIDER
    _SIM_TIME_PROVIDER = provider


def normalize_log_mode(raw: str) -> str:
    """Normalize CLI log mode text."""

    mode = str(raw).strip().upper()
    if mode not in _LEVEL_MAP:
        supported = ", ".join(_LEVEL_MAP.keys())
        raise ValueError(f"Invalid logger mode: {raw}. Supported values: {supported}.")
    logging.getLogger(__name__).debug("Normalized logger mode '%s' -> '%s'", raw, mode)
    return mode


def configure_logging(run_dir: Path, mode: str = "INFO") -> Path:
    """
    Configure root logger to write both to console and a run-local file.

    Returns path to the runtime log file.
    """

    normalized_mode = normalize_log_mode(mode)
    level = _LEVEL_MAP[normalized_mode]
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_path = run_dir / "runtime.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = SimulationTimeFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(runtime_log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)
    logging.getLogger(__name__).info(
        "Logger configured with mode=%s, file=%s",
        normalized_mode,
        runtime_log_path,
    )
    logging.getLogger(__name__).debug("Root logger handlers=%d", len(root_logger.handlers))
    return runtime_log_path

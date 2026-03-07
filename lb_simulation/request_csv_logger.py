"""CSV writer for per-request detail metrics."""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RequestCsvLogger:
    """Write one CSV row per completed request detail metrics."""

    FIELDS = [
        "rid",
        "class_id",
        "worker_id",
        "worker_class_id",
        "worker_service_model",
        "job_size",
        "model",
        "log_type",
        "t_arrival",
        "t_start",
        "t_done",
        "queue_len_on_dispatch",
        "n_local_at_start",
        "n_global_at_start",
        "lb_selected_worker_id",
        "routed_via_latency_tracker",
        "latency_tracked",
        "service_time",
        "latency",
        "lb_state",
        "lb_control_state",
        "queue_snapshot",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        logger.info("Opened detail metrics CSV at %s", self.path)

    def write(self, row: Dict[str, object]) -> None:
        if self._writer is None:
            raise RuntimeError("RequestCsvLogger is not opened.")
        self._writer.writerow(row)
        logger.debug("Wrote detail metric row rid=%s", row.get("rid"))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            logger.info("Closed detail metrics CSV at %s", self.path)
        self._file = None
        self._writer = None

"""Worker class configuration loading and runtime worker expansion."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .worker_models import WorkerServiceModel, create_worker_model

logger = logging.getLogger(__name__)


@dataclass
class WorkerClassSpec:
    """Configured worker class entry."""

    class_id: int
    count: int
    service_model: str
    queue_policy: str = "fcfs"
    queue_timeout_seconds: float | None = None
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerSpec:
    """Expanded worker instance used by the simulator."""

    worker_id: int
    worker_class_id: int
    worker_class_description: str
    service_model: str
    queue_policy: str
    queue_timeout_seconds: float | None
    service_model_impl: WorkerServiceModel


_QUEUE_POLICY_ALIASES = {
    "fcfs": "fcfs",
    "fifo": "fcfs",
    "sjf": "sjf",
    "shortest_job_first": "sjf",
    "shortest-job-first": "sjf",
    "shortest_request_first": "sjf",
    "shortest-request-first": "sjf",
    "short_request_first": "sjf",
    "short-request-first": "sjf",
}


def _parse_queue_policy(raw: object, class_idx: int) -> str:
    normalized = str(raw if raw is not None else "fcfs").strip().lower()
    if not normalized:
        normalized = "fcfs"
    queue_policy = _QUEUE_POLICY_ALIASES.get(normalized)
    if queue_policy is None:
        supported = ", ".join(sorted({"fcfs", "sjf"}))
        raise ValueError(
            f"classes[{class_idx}].queue_policy unsupported: {raw}. "
            f"Supported values: {supported}."
        )
    return queue_policy


def _parse_queue_timeout_seconds(raw: object, class_idx: int) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    if isinstance(raw, bool):
        raise ValueError(
            f"classes[{class_idx}].queue_timeout_seconds must be a number or null."
        )
    try:
        timeout_seconds = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"classes[{class_idx}].queue_timeout_seconds must be a number or null."
        ) from error
    if timeout_seconds < 0:
        raise ValueError(f"classes[{class_idx}].queue_timeout_seconds must be >= 0.")
    return timeout_seconds


def load_worker_class_config(path: Path) -> List[WorkerClassSpec]:
    """
    Load worker classes from JSON.

    Supported schema:
    {
      "classes": [
        {
          "class_id": 0,
          "description": "High-capacity GPU worker profile for heavy workloads",
          "count": 8,
          "service_model": "contention_lognormal",
          "queue_policy": "fcfs",
          "queue_timeout_seconds": 2.5,
          "params": {
            "a": 0.03,
            "b": 0.002,
            "c": 0.12,
            "d": 0.015,
            "n0": 32,
            "sigma": 0.2,
            "min_s": 0.001
          }
        },
        {
          "class_id": 1,
          "description": "Lightweight worker with linear service time by job_size",
          "count": 2,
          "service_model": "fixed",
          "queue_policy": "sjf",
          "params": {
            "service_time": 0.08
          }
        },
        {
          "class_id": 2,
          "description": "Worker using limited processor sharing",
          "count": 2,
          "service_model": "limited_processor_sharing",
          "queue_policy": "fcfs",
          "params": {
            "processing_rate": 120.0,
            "max_concurrency": 4
          }
        }
      ]
    }
    """

    logger.info("Loading worker class config from %s", path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        raw_classes = payload
    elif isinstance(payload, dict) and isinstance(payload.get("classes"), list):
        raw_classes = payload["classes"]
    else:
        raise ValueError("Worker class config must be a list or an object with 'classes'.")

    specs: List[WorkerClassSpec] = []
    seen_class_ids = set()

    for idx, item in enumerate(raw_classes):
        if not isinstance(item, dict):
            raise ValueError(f"classes[{idx}] must be an object.")

        class_id = int(item.get("class_id", idx))
        if class_id in seen_class_ids:
            raise ValueError(f"Duplicate worker class_id found: {class_id}")
        seen_class_ids.add(class_id)

        count_raw = item.get("count", item.get("workers", 0))
        count = int(count_raw)
        if count <= 0:
            raise ValueError(f"classes[{idx}].count must be > 0.")

        description = str(item.get("description", "")).strip()
        service_model = str(item.get("service_model", "contention_lognormal")).strip().lower()
        if not service_model:
            raise ValueError(f"classes[{idx}].service_model must be non-empty.")
        queue_policy = _parse_queue_policy(item.get("queue_policy", "fcfs"), idx)
        queue_timeout_seconds = _parse_queue_timeout_seconds(
            item.get("queue_timeout_seconds"),
            idx,
        )

        params = item.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"classes[{idx}].params must be an object.")

        specs.append(
            WorkerClassSpec(
                class_id=class_id,
                count=count,
                description=description,
                service_model=service_model,
                queue_policy=queue_policy,
                queue_timeout_seconds=queue_timeout_seconds,
                params=dict(params),
            )
        )
        logger.debug(
            (
                "Worker class loaded class_id=%d count=%d model=%s queue_policy=%s "
                "queue_timeout_seconds=%s description=%s"
            ),
            class_id,
            count,
            service_model,
            queue_policy,
            queue_timeout_seconds,
            description,
        )

    if not specs:
        raise ValueError("Worker class config does not contain any class entry.")

    logger.info("Worker class config loaded classes=%d", len(specs))
    return specs


def total_workers(worker_class_specs: Sequence[WorkerClassSpec]) -> int:
    """Compute total number of workers across all classes."""

    return sum(spec.count for spec in worker_class_specs)


def expand_worker_specs(worker_class_specs: Sequence[WorkerClassSpec]) -> List[WorkerSpec]:
    """Expand worker classes into one runtime worker spec per worker instance."""

    total = total_workers(worker_class_specs)
    if total <= 0:
        raise ValueError("Worker class config resolves to zero workers.")

    specs: List[WorkerSpec] = []
    next_worker_id = 0
    for class_spec in worker_class_specs:
        params: Mapping[str, Any] = class_spec.params
        for _ in range(class_spec.count):
            specs.append(
                WorkerSpec(
                    worker_id=next_worker_id,
                    worker_class_id=class_spec.class_id,
                    worker_class_description=class_spec.description,
                    service_model=class_spec.service_model,
                    queue_policy=class_spec.queue_policy,
                    queue_timeout_seconds=class_spec.queue_timeout_seconds,
                    service_model_impl=create_worker_model(
                        name=class_spec.service_model,
                        params=params,
                        total_workers=total,
                    ),
                )
            )
            next_worker_id += 1
    logger.info("Expanded worker specs total_workers=%d", len(specs))
    return specs

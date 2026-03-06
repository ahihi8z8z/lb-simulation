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
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerSpec:
    """Expanded worker instance used by the simulator."""

    worker_id: int
    worker_class_id: int
    service_model: str
    service_model_impl: WorkerServiceModel


def load_worker_class_config(path: Path) -> List[WorkerClassSpec]:
    """
    Load worker classes from JSON.

    Supported schema:
    {
      "classes": [
        {
          "class_id": 0,
          "count": 8,
          "service_model": "contention_lognormal",
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
          "count": 2,
          "service_model": "fixed",
          "params": {
            "service_time": 0.08
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

        service_model = str(item.get("service_model", "contention_lognormal")).strip().lower()
        if not service_model:
            raise ValueError(f"classes[{idx}].service_model must be non-empty.")

        params = item.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"classes[{idx}].params must be an object.")

        specs.append(
            WorkerClassSpec(
                class_id=class_id,
                count=count,
                service_model=service_model,
                params=dict(params),
            )
        )
        logger.debug(
            "Worker class loaded class_id=%d count=%d model=%s",
            class_id,
            count,
            service_model,
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
                    service_model=class_spec.service_model,
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

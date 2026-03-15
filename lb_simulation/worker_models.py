"""Pluggable worker service-time models and registry."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Type


@dataclass(frozen=True)
class ServiceTimeContext:
    """Runtime context used to sample one request service time."""

    job_size: int
    n_local: int
    n_global: int


class WorkerServiceModel(ABC):
    """Base class for worker-side service-time models."""

    name: str = ""

    @abstractmethod
    def sample_service_time(self, context: ServiceTimeContext, rng: random.Random) -> float:
        """Sample service time for one request."""


_WORKER_MODEL_REGISTRY: Dict[str, Type[WorkerServiceModel]] = {}
logger = logging.getLogger(__name__)


def register_worker_model(
    model_cls: Type[WorkerServiceModel],
) -> Type[WorkerServiceModel]:
    """Decorator used to register worker service-time models."""

    key = model_cls.name.strip().lower()
    if not key:
        raise ValueError("Worker model name must be a non-empty string.")
    if key in _WORKER_MODEL_REGISTRY:
        raise ValueError(f"Duplicate worker model registration: {key}")
    _WORKER_MODEL_REGISTRY[key] = model_cls
    logger.debug("Registered worker model: %s", key)
    return model_cls


def available_worker_models() -> List[str]:
    """Return names of all registered worker service models."""

    return list(_WORKER_MODEL_REGISTRY.keys())


def _as_float(params: Mapping[str, Any], key: str, default: float) -> float:
    value = params.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid float value for '{key}': {value}") from error


def _as_int(params: Mapping[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid int value for '{key}': {value}") from error


def create_worker_model(
    name: str,
    params: Mapping[str, Any],
    total_workers: int,
) -> WorkerServiceModel:
    """Instantiate one worker service model from config."""

    key = name.strip().lower()
    model_cls = _WORKER_MODEL_REGISTRY.get(key)
    if model_cls is None:
        supported = ", ".join(available_worker_models())
        raise ValueError(
            f"Unknown worker service model: {name}. Available models: {supported}"
        )
    logger.info("Creating worker model name=%s", key)
    model = model_cls(params=params, total_workers=total_workers)
    logger.debug("Worker model created name=%s params=%s", key, dict(params))
    return model


@register_worker_model
class ContentionLognormalModel(WorkerServiceModel):
    """S = (a + b*z) * (1 + c*n_local) * (1 + d*max(0,N-n0)) * LogNormal(0,sigma)."""

    name = "contention_lognormal"

    def __init__(self, params: Mapping[str, Any], total_workers: int) -> None:
        self.a = _as_float(params, "a", 0.03)
        self.b = _as_float(params, "b", 0.002)
        self.c = _as_float(params, "c", 0.12)
        self.d = _as_float(params, "d", 0.015)
        self.n0 = max(1, _as_int(params, "n0", max(1, total_workers * 4)))
        self.sigma = max(0.0, _as_float(params, "sigma", 0.20))
        self.min_s = max(1e-9, _as_float(params, "min_s", 0.001))

    def sample_service_time(self, context: ServiceTimeContext, rng: random.Random) -> float:
        base = self.a + self.b * context.job_size
        local_factor = 1.0 + self.c * context.n_local
        global_factor = 1.0 + self.d * max(0, context.n_global - self.n0)
        noise = rng.lognormvariate(0.0, self.sigma) if self.sigma > 0 else 1.0
        service = base * local_factor * global_factor * noise
        return max(self.min_s, service)


@register_worker_model
class LinearLognormalModel(WorkerServiceModel):
    """S = (a + b*z) * LogNormal(0,sigma)."""

    name = "linear_lognormal"

    def __init__(self, params: Mapping[str, Any], total_workers: int) -> None:
        del total_workers
        self.a = _as_float(params, "a", 0.03)
        self.b = _as_float(params, "b", 0.002)
        self.sigma = max(0.0, _as_float(params, "sigma", 0.20))
        self.min_s = max(1e-9, _as_float(params, "min_s", 0.001))

    def sample_service_time(self, context: ServiceTimeContext, rng: random.Random) -> float:
        base = self.a + self.b * context.job_size
        noise = rng.lognormvariate(0.0, self.sigma) if self.sigma > 0 else 1.0
        return max(self.min_s, base * noise)


@register_worker_model
class FixedServiceTimeModel(WorkerServiceModel):
    """S = service_time * job_size."""

    name = "fixed"

    def __init__(self, params: Mapping[str, Any], total_workers: int) -> None:
        del total_workers
        self.service_time = _as_float(params, "service_time", 0.05)
        if self.service_time < 0:
            raise ValueError("fixed.service_time must be >= 0.")

    def sample_service_time(self, context: ServiceTimeContext, rng: random.Random) -> float:
        del rng
        return context.job_size * self.service_time


@register_worker_model
class FixedLinearServiceTimeModel(WorkerServiceModel):
    """S = clip(a + b * job_size, min, max)."""

    name = "fixed_linear"

    def __init__(self, params: Mapping[str, Any], total_workers: int) -> None:
        del total_workers
        self.a = _as_float(params, "a", 0.03)
        self.b = _as_float(params, "b", 0.002)
        self.min_s = _as_float(params, "min", 1e-9)
        self.max_s = _as_float(params, "max", float("inf"))
        if self.min_s <= 0:
            raise ValueError("fixed_linear.min must be > 0.")

    def sample_service_time(self, context: ServiceTimeContext, rng: random.Random) -> float:
        del rng
        service_time = self.a + self.b * context.job_size
        service_time = min(self.max_s, max(self.min_s, service_time))
        return service_time


@register_worker_model
class LimitedProcessorSharingModel(WorkerServiceModel):
    """
    k-server approximation model used by InferencePool.

    InferencePool runs this model with `simpy.Resource(capacity=max_concurrency)`.
    `processing_rate` is interpreted as total worker rate and is evenly split
    across active slots in this approximation.
    """

    name = "limited_processor_sharing"

    def __init__(self, params: Mapping[str, Any], total_workers: int) -> None:
        del total_workers
        self.processing_rate = _as_float(params, "processing_rate", -1.0)
        self.max_concurrency = _as_int(params, "max_concurrency", -1)
        if self.processing_rate <= 0:
            raise ValueError("limited_processor_sharing.processing_rate must be > 0.")
        if self.max_concurrency <= 0:
            raise ValueError("limited_processor_sharing.max_concurrency must be > 0.")
        self.slot_processing_rate = self.processing_rate / float(self.max_concurrency)

    def sample_service_time(self, context: ServiceTimeContext, rng: random.Random) -> float:
        """
        Return isolated service time.

        Runtime executes this model as parallel servers (capacity=k), not exact
        processor sharing.
        """

        del rng
        work = max(0.0, float(context.job_size))
        return work / self.slot_processing_rate

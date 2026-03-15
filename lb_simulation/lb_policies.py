"""Pluggable load-balancing policies and policy registry."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Protocol, Sequence, Type

from .models import Request

logger = logging.getLogger(__name__)


class LoadBalancerView(Protocol):
    """State and helpers that policy implementations can access."""

    num_workers: int
    worker_ids: Sequence[int]
    lat_ewma: Sequence[float]
    inflight: Sequence[int]
    penalty: Sequence[float]
    feedback_count: Sequence[int]
    worker_weights: Sequence[float]
    explore_coef: float
    epsilon: float
    rng: random.Random

    def argmin_score(
        self,
        scores: Sequence[float],
        candidates: Sequence[int] | None = None,
    ) -> int:
        """Return index of minimum score with randomized tie-break."""


class LoadBalancingPolicy(ABC):
    """Base class for a load-balancing policy."""

    name: str = ""

    @abstractmethod
    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        """Choose a worker for this request."""


_POLICY_REGISTRY: Dict[str, Type[LoadBalancingPolicy]] = {}


def register_policy(
    policy_cls: Type[LoadBalancingPolicy],
) -> Type[LoadBalancingPolicy]:
    """Decorator used by policies to register themselves by name."""

    key = policy_cls.name.strip().lower()
    if not key:
        raise ValueError("Policy name must be a non-empty string.")
    if key in _POLICY_REGISTRY:
        raise ValueError(f"Duplicate policy registration: {key}")
    _POLICY_REGISTRY[key] = policy_cls
    logger.debug("Registered load-balancer policy: %s", key)
    return policy_cls


def available_policy_names() -> List[str]:
    """List names of all registered policies."""

    return list(_POLICY_REGISTRY.keys())


def create_policy(name: str) -> LoadBalancingPolicy:
    """Instantiate policy by name."""

    key = name.strip().lower()
    policy_cls = _POLICY_REGISTRY.get(key)
    if policy_cls is None:
        supported = ", ".join(available_policy_names())
        raise ValueError(f"Unknown policy: {name}. Available policies: {supported}")
    logger.info("Creating load-balancer policy: %s", key)
    policy = policy_cls()
    logger.debug("Policy instance created: %s", key)
    return policy


@register_policy
class RandomPolicy(LoadBalancingPolicy):
    name = "random"

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        return lb.rng.choice(list(lb.worker_ids))


class _BaseWeightedRoundRobinPolicy(LoadBalancingPolicy):
    """Common weighted round-robin selection logic for WRR-family policies."""

    def __init__(self) -> None:
        self._current_weights: List[float] = []

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        candidates = list(lb.worker_ids)
        if len(self._current_weights) != lb.num_workers:
            self._current_weights = [0.0 for _ in range(lb.num_workers)]

        total_weight = 0.0
        for idx in candidates:
            weight = max(1e-9, float(lb.worker_weights[idx]))
            self._current_weights[idx] += weight
            total_weight += weight

        max_value = max(self._current_weights[idx] for idx in candidates)
        candidates = [
            idx for idx in candidates if self._current_weights[idx] == max_value
        ]
        selected = lb.rng.choice(candidates)
        self._current_weights[selected] -= total_weight
        return selected


@register_policy
class StaticWrrPolicy(_BaseWeightedRoundRobinPolicy):
    name = "static-wrr"


@register_policy
class LpWrrPolicy(_BaseWeightedRoundRobinPolicy):
    name = "lp-wrr"


@register_policy
class SpWrrPolicy(_BaseWeightedRoundRobinPolicy):
    name = "sp-wrr"


@register_policy
class LeastConnectionPolicy(LoadBalancingPolicy):
    name = "least_connection"

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        return lb.argmin_score(lb.inflight)


@register_policy
class PowerOfTwoChoicesPolicy(LoadBalancingPolicy):
    name = "power_of_two_choices"

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        candidates = list(lb.worker_ids)
        if len(candidates) <= 1:
            return candidates[0]

        first, second = lb.rng.sample(candidates, k=2)
        first_load = lb.inflight[first]
        second_load = lb.inflight[second]
        if first_load < second_load:
            return first
        if second_load < first_load:
            return second
        return lb.rng.choice([first, second])


@register_policy
class MinEmaLatencyPolicy(LoadBalancingPolicy):
    name = "min_ema_latency"

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        return lb.argmin_score(lb.lat_ewma)


@register_policy
class LatencyP2CPolicy(LoadBalancingPolicy):
    name = "latency_p2c"

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        candidates = list(lb.worker_ids)
        if len(candidates) <= 1:
            return candidates[0]

        first, second = lb.rng.sample(candidates, k=2)
        first_latency = lb.lat_ewma[first]
        second_latency = lb.lat_ewma[second]
        if first_latency < second_latency:
            return first
        if second_latency < first_latency:
            return second
        return lb.rng.choice([first, second])

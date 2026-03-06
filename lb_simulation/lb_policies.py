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
    lat_ewma: Sequence[float]
    inflight: Sequence[int]
    penalty: Sequence[float]
    feedback_count: Sequence[int]
    worker_weights: Sequence[float]
    explore_coef: float
    epsilon: float
    rng: random.Random

    def argmin_score(self, scores: Sequence[float]) -> int:
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
        return lb.rng.randrange(lb.num_workers)


@register_policy
class RoundRobinPolicy(LoadBalancingPolicy):
    name = "round_robin"

    def __init__(self) -> None:
        self._next_worker = 0

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        worker_id = self._next_worker
        self._next_worker = (self._next_worker + 1) % lb.num_workers
        return worker_id


@register_policy
class WeightedRoundRobinPolicy(LoadBalancingPolicy):
    name = "weighted_round_robin"

    def __init__(self) -> None:
        self._current_weights: List[float] = []

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        if len(self._current_weights) != lb.num_workers:
            self._current_weights = [0.0 for _ in range(lb.num_workers)]

        total_weight = 0.0
        for idx in range(lb.num_workers):
            weight = max(1e-9, float(lb.worker_weights[idx]))
            self._current_weights[idx] += weight
            total_weight += weight

        max_value = max(self._current_weights)
        candidates = [
            idx for idx, value in enumerate(self._current_weights) if value == max_value
        ]
        selected = lb.rng.choice(candidates)
        self._current_weights[selected] -= total_weight
        return selected


@register_policy
class LeastInflightPolicy(LoadBalancingPolicy):
    name = "least_inflight"

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        return lb.argmin_score(lb.inflight)


@register_policy
class PeakEwmaPolicy(LoadBalancingPolicy):
    name = "peak_ewma"

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        scores = [
            lb.lat_ewma[i] * (1.0 + lb.inflight[i]) + lb.penalty[i]
            for i in range(lb.num_workers)
        ]
        return lb.argmin_score(scores)


@register_policy
class LatencyOnlyPolicy(LoadBalancingPolicy):
    name = "latency_only"

    def choose_worker(self, request: Request, lb: LoadBalancerView) -> int:
        del request
        # Epsilon-greedy keeps low-cost exploration active.
        if lb.rng.random() < lb.epsilon:
            return lb.rng.randrange(lb.num_workers)

        # Optimism bonus makes under-sampled workers more likely to be tried.
        scores = []
        for i in range(lb.num_workers):
            base = lb.lat_ewma[i] * (1.0 + lb.inflight[i]) + lb.penalty[i]
            bonus = lb.explore_coef / ((lb.feedback_count[i] + 1) ** 0.5)
            scores.append(base - bonus)
        return lb.argmin_score(scores)

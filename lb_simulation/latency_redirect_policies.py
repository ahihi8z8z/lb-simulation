"""Pluggable latency-tracker redirect policies."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Type

from .models import Request


class LatencyRedirectPolicy(ABC):
    """Base class for redirect policies used by latency tracker."""

    name: str = ""
    forward_mode: str = "round_robin"
    rate: float = 0.0

    @abstractmethod
    def should_redirect(self, request: Request, rng: random.Random) -> bool:
        """Return whether this request should be redirected via latency tracker."""


_REDIRECT_POLICY_REGISTRY: Dict[str, Type[LatencyRedirectPolicy]] = {}


def register_latency_redirect_policy(
    policy_cls: Type[LatencyRedirectPolicy],
) -> Type[LatencyRedirectPolicy]:
    """Decorator used by policies to register themselves by name."""

    key = policy_cls.name.strip().lower()
    if not key:
        raise ValueError("Latency redirect policy name must be a non-empty string.")
    if key in _REDIRECT_POLICY_REGISTRY:
        raise ValueError(f"Duplicate latency redirect policy registration: {key}")
    _REDIRECT_POLICY_REGISTRY[key] = policy_cls
    return policy_cls


def available_latency_redirect_policies() -> List[str]:
    """Return all registered latency redirect policy names."""

    return list(_REDIRECT_POLICY_REGISTRY.keys())


def create_latency_redirect_policy(
    name: str,
    params: Mapping[str, Any],
) -> LatencyRedirectPolicy:
    """Instantiate one latency redirect policy by name."""

    key = name.strip().lower()
    policy_cls = _REDIRECT_POLICY_REGISTRY.get(key)
    if policy_cls is None:
        supported = ", ".join(available_latency_redirect_policies())
        raise ValueError(
            f"Unknown latency redirect policy: {name}. Available policies: {supported}"
        )
    return policy_cls(params=params)


def _as_float(params: Mapping[str, Any], key: str, default: float) -> float:
    value = params.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid float value for '{key}': {value}") from error


@register_latency_redirect_policy
class FixedRateRedirectPolicy(LatencyRedirectPolicy):
    """Redirect a fixed fraction of requests to latency tracker."""

    name = "fixed_rate"
    forward_mode = "round_robin"

    def __init__(self, params: Mapping[str, Any]) -> None:
        rate = _as_float(params, "rate", 0.05)
        if not (0.0 <= rate <= 1.0):
            raise ValueError("redirect policy fixed_rate requires 0 <= rate <= 1.")
        self.rate = rate

    def should_redirect(self, request: Request, rng: random.Random) -> bool:
        del request
        return rng.random() < self.rate


@register_latency_redirect_policy
class TrackAllRedirectPolicy(LatencyRedirectPolicy):
    """
    Track latency of all requests.

    Semantics:
    - Decision-stage redirect rate is 0: tracker is not part of worker scoring.
    - Post-decision redirect rate is 1: every request is routed via tracker.
    - Tracker forwards to exactly the worker selected by LB policy.
    """

    name = "track_all"
    forward_mode = "selected_worker"
    rate = 1.0

    def __init__(self, params: Mapping[str, Any]) -> None:
        del params

    def should_redirect(self, request: Request, rng: random.Random) -> bool:
        del request, rng
        return True

"""Controller module for LB parameter control and sampled latency tracking."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .models import Request

LATENCY_AWARE_POLICIES = frozenset({"peak_ewma", "latency_only"})


@dataclass
class LatencyTrackerConfig:
    """Configuration for sampled latency tracking."""

    enabled: Optional[bool] = None
    sample_rate: float = 0.05
    init_estimate: float = 0.5
    ewma_gamma: float = 0.10


@dataclass
class WrrControlConfig:
    """Configuration for weighted-round-robin weight control."""

    mode: str = "none"
    weights: Optional[List[float]] = None
    update_every_samples: int = 20
    inverse_power: float = 1.0
    min_weight: float = 0.1
    max_weight: float = 10.0


@dataclass
class ControllerConfig:
    """Top-level controller configuration."""

    mode: str = "none"
    latency_tracker: LatencyTrackerConfig = field(default_factory=LatencyTrackerConfig)
    wrr: WrrControlConfig = field(default_factory=WrrControlConfig)


def _to_float(raw: object, key: str, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid float value for {key}: {raw}") from error


def _to_int(raw: object, key: str, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid int value for {key}: {raw}") from error


def _to_bool_optional(raw: object, key: str, default: Optional[bool]) -> Optional[bool]:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Invalid bool value for {key}: {raw}")


def _parse_controller_payload(
    payload: Dict[str, object],
) -> ControllerConfig:
    latency_cfg_raw = payload.get("latency_tracker")
    latency_cfg_dict = latency_cfg_raw if isinstance(latency_cfg_raw, dict) else {}
    latency_cfg = LatencyTrackerConfig(
        enabled=_to_bool_optional(
            latency_cfg_dict.get("enabled"), "latency_tracker.enabled", None
        ),
        sample_rate=max(
            0.0,
            min(
                1.0,
                _to_float(
                    latency_cfg_dict.get("sample_rate"),
                    "latency_tracker.sample_rate",
                    0.05,
                ),
            ),
        ),
        init_estimate=max(
            1e-9,
            _to_float(
                latency_cfg_dict.get("init_estimate"),
                "latency_tracker.init_estimate",
                0.5,
            ),
        ),
        ewma_gamma=max(
            0.0,
            min(
                1.0,
                _to_float(
                    latency_cfg_dict.get("ewma_gamma"),
                    "latency_tracker.ewma_gamma",
                    0.10,
                ),
            ),
        ),
    )

    wrr_cfg_raw = payload.get("wrr")
    wrr_cfg_dict = wrr_cfg_raw if isinstance(wrr_cfg_raw, dict) else {}
    weights_raw = wrr_cfg_dict.get("weights")
    weights: Optional[List[float]] = None
    if weights_raw is not None:
        if not isinstance(weights_raw, list):
            raise ValueError("wrr.weights must be a list when provided.")
        weights = []
        for idx, value in enumerate(weights_raw):
            weight = _to_float(value, f"wrr.weights[{idx}]", 1.0)
            if weight <= 0:
                raise ValueError(f"wrr.weights[{idx}] must be > 0.")
            weights.append(weight)

    wrr_cfg = WrrControlConfig(
        mode=str(wrr_cfg_dict.get("mode", "none")).strip().lower() or "none",
        weights=weights,
        update_every_samples=max(
            1,
            _to_int(
                wrr_cfg_dict.get("update_every_samples"),
                "wrr.update_every_samples",
                20,
            ),
        ),
        inverse_power=max(
            1e-9,
            _to_float(
                wrr_cfg_dict.get("inverse_power"),
                "wrr.inverse_power",
                1.0,
            ),
        ),
        min_weight=max(
            1e-9,
            _to_float(
                wrr_cfg_dict.get("min_weight"),
                "wrr.min_weight",
                0.1,
            ),
        ),
        max_weight=max(
            1e-9,
            _to_float(
                wrr_cfg_dict.get("max_weight"),
                "wrr.max_weight",
                10.0,
            ),
        ),
    )

    if wrr_cfg.max_weight < wrr_cfg.min_weight:
        raise ValueError("wrr.max_weight must be >= wrr.min_weight.")
    if wrr_cfg.mode not in {"none", "inverse_latency"}:
        raise ValueError("wrr.mode must be one of: none, inverse_latency.")

    return ControllerConfig(
        mode=str(payload.get("mode", "none")).strip().lower() or "none",
        latency_tracker=latency_cfg,
        wrr=wrr_cfg,
    )


def load_controller_config(
    path: Optional[Path],
) -> ControllerConfig:
    """Load controller config from optional JSON path."""

    if path is None:
        return ControllerConfig()

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Controller config must be a JSON object.")
    return _parse_controller_payload(payload)


class SampledLatencyTracker:
    """Track latency estimates from a sampled subset of request completions."""

    def __init__(
        self,
        num_workers: int,
        config: LatencyTrackerConfig,
        ewma_gamma: float,
    ) -> None:
        self.sample_rate = config.sample_rate
        self.ewma_gamma = max(0.0, min(1.0, float(ewma_gamma)))
        self.estimates: List[float] = [config.init_estimate for _ in range(num_workers)]
        self.sample_counts: List[int] = [0 for _ in range(num_workers)]
        self.sampled_requests = 0

    def should_track(self, rng: random.Random) -> bool:
        return rng.random() < self.sample_rate

    def observe(self, worker_id: int, latency: float) -> None:
        previous = self.estimates[worker_id]
        gamma = self.ewma_gamma
        estimate = (1.0 - gamma) * previous + gamma * latency
        self.estimates[worker_id] = max(1e-9, estimate)
        self.sample_counts[worker_id] += 1
        self.sampled_requests += 1


class LoadBalancerController:
    """Controller hooks for sampled latency tracking and LB parameter control."""

    def __init__(
        self,
        policy: str,
        num_workers: int,
        config: ControllerConfig,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.policy = policy.strip().lower()
        self.num_workers = num_workers
        self.config = config
        self.rng = rng or random.Random()
        self.control_mode = config.mode
        self.completion_count = 0
        self.track_decision_count = 0

        requested = config.latency_tracker.enabled
        if requested is None:
            requested = self.policy in LATENCY_AWARE_POLICIES
        self.latency_tracker_enabled = requested

        if (self.policy in LATENCY_AWARE_POLICIES) and (not self.latency_tracker_enabled):
            raise ValueError(
                f"Policy '{self.policy}' requires latency tracker in controller."
            )

        self.latency_tracker: Optional[SampledLatencyTracker]
        if self.latency_tracker_enabled:
            self.latency_tracker = SampledLatencyTracker(
                num_workers=num_workers,
                config=config.latency_tracker,
                ewma_gamma=config.latency_tracker.ewma_gamma,
            )
        else:
            self.latency_tracker = None

    def initialize(self, lb: "LoadBalancer") -> None:
        if self.latency_tracker is not None:
            for worker_id, estimate in enumerate(self.latency_tracker.estimates):
                lb.set_latency_estimate(worker_id, estimate, feedback_count=0)

        if self.config.wrr.weights is not None:
            lb.set_worker_weights(self.config.wrr.weights)

    def should_track_latency(self, request: Request, worker_id: int) -> bool:
        del request, worker_id
        self.track_decision_count += 1
        if self.latency_tracker is None:
            return False
        return self.latency_tracker.should_track(self.rng)

    def _maybe_update_wrr_weights(self, lb: "LoadBalancer") -> None:
        if self.policy != "weighted_round_robin":
            return
        if self.config.wrr.mode != "inverse_latency":
            return
        if self.latency_tracker is None:
            return
        if self.latency_tracker.sampled_requests <= 0:
            return
        if (
            self.latency_tracker.sampled_requests % self.config.wrr.update_every_samples
            != 0
        ):
            return

        weights: List[float] = []
        for estimate in self.latency_tracker.estimates:
            weight = (1.0 / max(1e-9, estimate)) ** self.config.wrr.inverse_power
            weight = min(max(weight, self.config.wrr.min_weight), self.config.wrr.max_weight)
            weights.append(weight)
        lb.set_worker_weights(weights)

    def on_request_complete(
        self,
        request: Request,
        worker_id: int,
        latency: float,
        latency_tracked: bool,
        lb: "LoadBalancer",
    ) -> None:
        del request
        self.completion_count += 1
        if latency_tracked and self.latency_tracker is not None:
            self.latency_tracker.observe(worker_id, latency)
            lb.set_latency_estimate(
                worker_id,
                self.latency_tracker.estimates[worker_id],
                self.latency_tracker.sample_counts[worker_id],
            )
            self._maybe_update_wrr_weights(lb)

    def summarize(self, lb: "LoadBalancer") -> Dict[str, object]:
        summary: Dict[str, object] = {
            "mode": self.control_mode,
            "policy": self.policy,
            "track_decisions": self.track_decision_count,
            "completions_seen": self.completion_count,
            "wrr_control_mode": self.config.wrr.mode,
            "wrr_weights": [float(value) for value in lb.worker_weights],
            "latency_tracker_enabled": self.latency_tracker is not None,
        }

        if self.latency_tracker is not None:
            summary["latency_sample_rate"] = self.latency_tracker.sample_rate
            summary["latency_tracker_ewma_gamma"] = self.latency_tracker.ewma_gamma
            summary["latency_samples_total"] = self.latency_tracker.sampled_requests
            summary["latency_samples_by_worker"] = list(self.latency_tracker.sample_counts)
            summary["latency_estimate_by_worker"] = list(self.latency_tracker.estimates)
        else:
            summary["latency_sample_rate"] = 0.0
            summary["latency_tracker_ewma_gamma"] = 0.0
            summary["latency_samples_total"] = 0
            summary["latency_samples_by_worker"] = []
            summary["latency_estimate_by_worker"] = []

        return summary

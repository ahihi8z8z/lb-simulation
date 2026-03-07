"""Controller module for LB parameter control and sampled latency tracking."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .lb_control_modules import WrrLpControlParams, create_load_balancer_control_module
from .latency_tracker import LatencyTrackerWorker
from .models import Request

LATENCY_AWARE_POLICIES = frozenset({"latency_only"})
logger = logging.getLogger(__name__)


@dataclass
class LatencyRedirectPolicyConfig:
    """Configuration for latency-tracker redirect policy."""

    name: str = "fixed_rate"
    params: Dict[str, object] = field(default_factory=lambda: {"rate": 0.05})


@dataclass
class LatencyTrackerConfig:
    """Configuration for latency tracker worker."""

    enabled: Optional[bool] = None
    init_estimate: float = 0.5
    ewma_gamma: float = 0.10
    redirect_policy: LatencyRedirectPolicyConfig = field(
        default_factory=LatencyRedirectPolicyConfig
    )


@dataclass
class WrrControlConfig:
    """Configuration for weighted-round-robin weight control."""

    mode: str = "none"
    weights: Optional[List[float]] = None
    update_interval_seconds: float = 60.0
    min_weight: float = 0.1
    max_weight: float = 10.0
    lp_balance_tolerance: float = 0.25
    lp_ewma_gamma: float = 0.10


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
    redirect_policy_cfg = _parse_redirect_policy_config(
        raw=latency_cfg_dict.get("redirect_policy"),
        legacy_sample_rate_raw=latency_cfg_dict.get("sample_rate"),
    )
    latency_cfg = LatencyTrackerConfig(
        enabled=_to_bool_optional(
            latency_cfg_dict.get("enabled"), "latency_tracker.enabled", None
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
        redirect_policy=redirect_policy_cfg,
    )

    wrr_cfg_raw = payload.get("wrr")
    wrr_cfg_dict = wrr_cfg_raw if isinstance(wrr_cfg_raw, dict) else {}
    interval_raw = wrr_cfg_dict.get("update_interval_seconds")
    if interval_raw is None:
        interval_raw = wrr_cfg_dict.get("update_every_samples")
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
        update_interval_seconds=max(
            1e-6,
            _to_float(
                interval_raw,
                "wrr.update_interval_seconds",
                60.0,
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
        lp_balance_tolerance=max(
            0.0,
            _to_float(
                wrr_cfg_dict.get("lp_balance_tolerance"),
                "wrr.lp_balance_tolerance",
                0.25,
            ),
        ),
        lp_ewma_gamma=max(
            0.0,
            min(
                1.0,
                _to_float(
                    wrr_cfg_dict.get("lp_ewma_gamma"),
                    "wrr.lp_ewma_gamma",
                    0.10,
                ),
            ),
        ),
    )

    if wrr_cfg.max_weight < wrr_cfg.min_weight:
        raise ValueError("wrr.max_weight must be >= wrr.min_weight.")
    if wrr_cfg.mode not in {"none", "lp_latency", "separate_lp"}:
        raise ValueError("wrr.mode must be one of: none, lp_latency, separate_lp.")
    if (wrr_cfg.mode in {"lp_latency", "separate_lp"}) and latency_cfg.enabled is not True:
        raise ValueError(
            "wrr.mode in {'lp_latency','separate_lp'} requires "
            "latency_tracker.enabled=true in controller config."
        )

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
        logger.info("Using default controller config")
        return ControllerConfig()

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Controller config must be a JSON object.")
    config = _parse_controller_payload(payload)
    logger.info("Loaded controller config from %s", path)
    logger.debug("Controller config payload: %s", payload)
    return config


def _parse_redirect_policy_config(
    raw: object,
    legacy_sample_rate_raw: object,
) -> LatencyRedirectPolicyConfig:
    """Parse latency-tracker redirect policy with backward compatibility."""

    if raw is None:
        rate = max(
            0.0,
            min(
                1.0,
                _to_float(
                    legacy_sample_rate_raw,
                    "latency_tracker.sample_rate",
                    0.05,
                ),
            ),
        )
        return LatencyRedirectPolicyConfig(
            name="fixed_rate",
            params={"rate": rate},
        )

    if isinstance(raw, str):
        name = raw.strip().lower() or "fixed_rate"
        return LatencyRedirectPolicyConfig(name=name, params={})

    if not isinstance(raw, dict):
        raise ValueError(
            "latency_tracker.redirect_policy must be a string or object."
        )

    name = str(raw.get("name", "fixed_rate")).strip().lower() or "fixed_rate"
    params_raw = raw.get("params", {})
    if params_raw is None:
        params_raw = {}
    if not isinstance(params_raw, dict):
        raise ValueError("latency_tracker.redirect_policy.params must be an object.")

    params: Dict[str, object] = dict(params_raw)
    # Convenience form:
    # "redirect_policy": {"name": "fixed_rate", "rate": 0.05}
    if ("rate" in raw) and ("rate" not in params):
        params["rate"] = raw.get("rate")
    return LatencyRedirectPolicyConfig(name=name, params=params)


class LoadBalancerController:
    """Controller hooks for latency tracking and LB parameter control."""

    def __init__(
        self,
        policy: str,
        num_workers: int,
        config: ControllerConfig,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.policy = policy.strip().lower()
        self.num_workers = num_workers
        self.tracker_worker_id = num_workers
        self.config = config
        self.rng = rng or random.Random()
        self.control_mode = config.mode
        self.completion_count = 0
        self.class_load_balancers: Dict[int, "LoadBalancer"] = {}
        self.latency_trackers_by_class: Dict[int, LatencyTrackerWorker] = {}

        requested = config.latency_tracker.enabled
        self.latency_tracker_enabled = bool(requested)

        if (self.policy in LATENCY_AWARE_POLICIES) and (not self.latency_tracker_enabled):
            raise ValueError(
                f"Policy '{self.policy}' requires latency_tracker.enabled=true in controller config."
            )
        if self.config.wrr.mode in {"lp_latency", "separate_lp"} and (
            self.policy != "weighted_round_robin"
        ):
            raise ValueError(
                "wrr.mode in {'lp_latency','separate_lp'} requires "
                "policy='weighted_round_robin'."
            )
        if (
            self.policy == "weighted_round_robin"
            and self.config.wrr.mode in {"lp_latency", "separate_lp"}
            and (not self.latency_tracker_enabled)
        ):
            raise ValueError(
                "Policy 'weighted_round_robin' with wrr.mode in "
                "{'lp_latency','separate_lp'} requires latency_tracker.enabled=true "
                "in controller config."
            )

        if (self.policy == "weighted_round_robin") and (
            self.config.wrr.mode in {"lp_latency", "separate_lp"}
        ):
            module_name = (
                "wrr_lp_latency"
                if self.config.wrr.mode == "lp_latency"
                else "wrr_separate_lp_latency"
            )
            self.lb_control_module = create_load_balancer_control_module(
                module_name,
                num_workers=self.num_workers,
                params=WrrLpControlParams(
                    update_interval_seconds=self.config.wrr.update_interval_seconds,
                    min_weight=self.config.wrr.min_weight,
                    max_weight=self.config.wrr.max_weight,
                    lp_balance_tolerance=self.config.wrr.lp_balance_tolerance,
                    lp_ewma_gamma=self.config.wrr.lp_ewma_gamma,
                ),
            )
        else:
            self.lb_control_module = create_load_balancer_control_module(
                "none",
                num_workers=self.num_workers,
            )
        logger.info(
            "LoadBalancerController initialized policy=%s tracker_enabled=%s lb_control=%s",
            self.policy,
            self.latency_tracker_enabled,
            self.lb_control_module.name,
        )

    def initialize(self, lbs_by_class: Mapping[int, "LoadBalancer"]) -> None:
        self.class_load_balancers = {
            int(class_id): lb for class_id, lb in lbs_by_class.items()
        }
        self.latency_trackers_by_class = {}
        if not self.class_load_balancers:
            raise ValueError("Controller requires at least one load balancer instance.")

        for class_id, lb in self.class_load_balancers.items():
            class_tracker: Optional[LatencyTrackerWorker] = None
            if self.latency_tracker_enabled:
                class_tracker = LatencyTrackerWorker(
                    num_workers=self.num_workers,
                    tracker_worker_id=self.tracker_worker_id,
                    config=self.config.latency_tracker,
                    rng=random.Random(self.rng.randrange(1, 2**31)),
                )
                self.latency_trackers_by_class[class_id] = class_tracker
                lb.configure_latency_tracker(
                    tracker_worker_id=class_tracker.tracker_worker_id,
                    should_redirect=class_tracker.should_redirect,
                )
                for worker_id, estimate in enumerate(class_tracker.estimates):
                    lb.set_latency_estimate(worker_id, estimate, feedback_count=0)

            if self.config.wrr.weights is not None:
                lb.set_worker_weights(self.config.wrr.weights)

        self.lb_control_module.initialize(self.class_load_balancers)
        logger.info(
            "Controller initialized into %d load balancers",
            len(self.class_load_balancers),
        )

    def is_latency_tracker_worker(self, worker_id: int) -> bool:
        return self.latency_tracker_enabled and (worker_id == self.tracker_worker_id)

    def forward_via_latency_tracker(
        self,
        request: Request,
        selected_worker_id: Optional[int] = None,
    ) -> int:
        class_id = int(request.class_id)
        class_tracker = self.latency_trackers_by_class.get(class_id)
        if class_tracker is None:
            raise RuntimeError(
                f"Latency tracker is not enabled for class_id={class_id}."
            )
        worker_id = class_tracker.pick_forward_worker(
            request,
            selected_worker_id=selected_worker_id,
        )
        logger.debug(
            "Latency tracker forwarded rid=%d -> worker=%d",
            request.rid,
            worker_id,
        )
        return worker_id

    def on_request_complete(
        self,
        request: Request,
        worker_id: int,
        latency: float,
        latency_tracked: bool,
    ) -> None:
        self.completion_count += 1
        class_id = int(request.class_id)
        class_lb = self.class_load_balancers.get(class_id)
        if class_lb is None:
            raise RuntimeError(f"No load balancer found for class_id={class_id}.")
        class_tracker = self.latency_trackers_by_class.get(class_id)
        if latency_tracked and (class_tracker is not None):
            class_tracker.observe(worker_id, latency)
            class_lb.set_latency_estimate(
                worker_id,
                class_tracker.estimates[worker_id],
                class_tracker.sample_counts[worker_id],
            )
        self.lb_control_module.on_request_complete(
            request=request,
            worker_id=worker_id,
            latency=latency,
            latency_tracked=latency_tracked,
        )
        logger.debug(
            "Completion processed rid=%d worker=%d tracked=%s latency=%.4f",
            request.rid,
            worker_id,
            latency_tracked,
            latency,
        )

    def summarize(self) -> Dict[str, object]:
        sorted_lb_items = sorted(self.class_load_balancers.items(), key=lambda item: item[0])
        sorted_tracker_items = sorted(
            self.latency_trackers_by_class.items(), key=lambda item: item[0]
        )
        if sorted_lb_items:
            _, first_lb = sorted_lb_items[0]
            legacy_wrr_weights = [float(value) for value in first_lb.worker_weights]
        else:
            legacy_wrr_weights = []
        summary: Dict[str, object] = {
            "mode": self.control_mode,
            "policy": self.policy,
            "completions_seen": self.completion_count,
            "lb_count": len(sorted_lb_items),
            "wrr_control_mode": self.config.wrr.mode,
            "lb_control_module": self.lb_control_module.name,
            "wrr_weights": legacy_wrr_weights,
            "wrr_weights_by_class": {
                str(class_id): [float(value) for value in lb.worker_weights]
                for class_id, lb in sorted_lb_items
            },
            "latency_tracker_enabled": self.latency_tracker_enabled,
        }
        summary.update(self.lb_control_module.summarize(self.class_load_balancers))

        if sorted_tracker_items:
            track_decisions_total = sum(
                tracker.redirect_decisions for _, tracker in sorted_tracker_items
            )
            track_redirected_total = sum(
                tracker.redirected_requests for _, tracker in sorted_tracker_items
            )
            latency_samples_total = sum(
                tracker.sampled_requests for _, tracker in sorted_tracker_items
            )
            track_decisions_by_class = {
                str(class_id): int(tracker.redirect_decisions)
                for class_id, tracker in sorted_tracker_items
            }
            track_redirected_by_class = {
                str(class_id): int(tracker.redirected_requests)
                for class_id, tracker in sorted_tracker_items
            }
            summary["track_decisions"] = track_decisions_total
            summary["track_redirected"] = track_redirected_total
            summary["track_decisions_by_class"] = track_decisions_by_class
            summary["track_redirected_by_class"] = track_redirected_by_class
            summary["latency_sample_rate"] = float(sorted_tracker_items[0][1].redirect_rate)
            summary["latency_tracker_ewma_gamma"] = float(sorted_tracker_items[0][1].ewma_gamma)
            summary["latency_tracker_worker_id"] = sorted_tracker_items[0][1].tracker_worker_id
            summary["latency_tracker_dispatches"] = sum(
                lb.latency_tracker_dispatches for _, lb in sorted_lb_items
            )
            summary["latency_tracker_inflight"] = sum(
                lb.latency_tracker_inflight for _, lb in sorted_lb_items
            )
            summary["latency_redirect_policy"] = {
                "name": sorted_tracker_items[0][1].redirect_policy_name,
                "params": dict(sorted_tracker_items[0][1].redirect_policy_params),
            }
            summary["latency_samples_total"] = latency_samples_total
            sample_counts_by_class = {
                str(class_id): list(tracker.sample_counts)
                for class_id, tracker in sorted_tracker_items
            }
            estimates_by_class = {
                str(class_id): list(tracker.estimates)
                for class_id, tracker in sorted_tracker_items
            }
            summary["latency_samples_by_worker_and_class"] = sample_counts_by_class
            summary["latency_estimate_by_worker_and_class"] = estimates_by_class
            aggregated_samples = [0 for _ in range(self.num_workers)]
            aggregated_estimates = [0.0 for _ in range(self.num_workers)]
            for _, tracker in sorted_tracker_items:
                for worker_id in range(self.num_workers):
                    aggregated_samples[worker_id] += int(tracker.sample_counts[worker_id])
                    aggregated_estimates[worker_id] += float(tracker.estimates[worker_id])
            tracker_count = float(len(sorted_tracker_items))
            summary["latency_samples_by_worker"] = aggregated_samples
            summary["latency_estimate_by_worker"] = [
                value / tracker_count for value in aggregated_estimates
            ]
        else:
            summary["track_decisions"] = 0
            summary["track_redirected"] = 0
            summary["track_decisions_by_class"] = {}
            summary["track_redirected_by_class"] = {}
            summary["latency_sample_rate"] = 0.0
            summary["latency_tracker_ewma_gamma"] = 0.0
            summary["latency_tracker_worker_id"] = None
            summary["latency_tracker_dispatches"] = 0
            summary["latency_tracker_inflight"] = 0
            summary["latency_redirect_policy"] = {}
            summary["latency_samples_total"] = 0
            summary["latency_samples_by_worker"] = []
            summary["latency_estimate_by_worker"] = []
            summary["latency_samples_by_worker_and_class"] = {}
            summary["latency_estimate_by_worker_and_class"] = {}

        logger.info("Controller summary generated")
        return summary

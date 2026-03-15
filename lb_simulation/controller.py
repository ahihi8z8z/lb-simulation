"""Controller module for LB parameter control and sampled latency tracking."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from .lb_control_modules import WrrLpControlParams, create_load_balancer_control_module
from .latency_tracker import LatencyTrackerWorker
from .models import Request

LATENCY_AWARE_POLICIES = frozenset(
    {
        "min_ema_latency",
        "latency_p2c",
    }
)
TRACKER_REQUIRED_POLICIES = frozenset(
    set(LATENCY_AWARE_POLICIES) | {"lp-wrr", "sp-wrr"}
)
WRR_POLICIES = frozenset({"static-wrr", "lp-wrr", "sp-wrr"})
logger = logging.getLogger(__name__)


@dataclass
class LatencyRedirectPolicyConfig:
    """Configuration for latency-tracker redirect policy."""

    name: str = "fixed_rate"
    params: Dict[str, object] = field(default_factory=lambda: {"rate": 0.05})


@dataclass
class LatencyTrackerConfig:
    """Configuration for latency tracker worker."""

    init_estimate: float = 0.5
    ewma_gamma: float = 0.10
    redirect_policy: LatencyRedirectPolicyConfig = field(
        default_factory=LatencyRedirectPolicyConfig
    )


@dataclass
class LpWrrControlConfig:
    """Configuration for lp-wrr weight control."""

    update_interval_seconds: float = 60.0
    min_weight: float = 0.1
    max_weight: float = 10.0
    lp_balance_tolerance: float = 0.25
    lp_ewma_gamma: float = 0.10


@dataclass
class SpWrrControlConfig:
    """Configuration for sp-wrr weight control."""

    update_interval_seconds: float = 60.0
    min_weight: float = 0.1
    max_weight: float = 10.0
    lp_balance_tolerance: float = 0.25
    lp_ewma_gamma: float = 0.10


@dataclass
class ControllerConfig:
    """Top-level controller configuration."""

    latency_tracker: Optional[LatencyTrackerConfig] = None
    lp_wrr: Optional[LpWrrControlConfig] = None
    sp_wrr: Optional[SpWrrControlConfig] = None


def _to_float(raw: object, key: str, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid float value for {key}: {raw}") from error


def _parse_controller_payload(
    payload: Dict[str, object],
) -> ControllerConfig:
    latency_cfg: Optional[LatencyTrackerConfig] = None
    if "latency_tracker" in payload:
        latency_cfg_raw = payload.get("latency_tracker")
        if not isinstance(latency_cfg_raw, dict):
            raise ValueError("controller.latency_tracker must be an object when provided.")
        latency_cfg_dict = latency_cfg_raw
        redirect_policy_cfg = _parse_redirect_policy_config(
            raw=latency_cfg_dict.get("redirect_policy"),
            legacy_sample_rate_raw=latency_cfg_dict.get("sample_rate"),
        )
        latency_cfg = LatencyTrackerConfig(
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

    if "wrr" in payload:
        raise ValueError(
            "controller.wrr is removed. Use controller.lp-wrr or controller.sp-wrr."
        )

    def _parse_wrr_control_block(
        raw: object,
        key_prefix: str,
    ) -> Tuple[float, float, float, float, float]:
        if raw is None:
            raise ValueError(f"controller.{key_prefix} must be an object when provided.")
        if not isinstance(raw, dict):
            raise ValueError(f"controller.{key_prefix} must be an object when provided.")
        update_interval_seconds = max(
            1e-6,
            _to_float(
                raw.get("update_interval_seconds"),
                f"{key_prefix}.update_interval_seconds",
                60.0,
            ),
        )
        min_weight = max(
            1e-9,
            _to_float(
                raw.get("min_weight"),
                f"{key_prefix}.min_weight",
                0.1,
            ),
        )
        max_weight = max(
            1e-9,
            _to_float(
                raw.get("max_weight"),
                f"{key_prefix}.max_weight",
                10.0,
            ),
        )
        lp_balance_tolerance = max(
            0.0,
            _to_float(
                raw.get("lp_balance_tolerance"),
                f"{key_prefix}.lp_balance_tolerance",
                0.25,
            ),
        )
        lp_ewma_gamma = max(
            0.0,
            min(
                1.0,
                _to_float(
                    raw.get("lp_ewma_gamma"),
                    f"{key_prefix}.lp_ewma_gamma",
                    0.10,
                ),
            ),
        )
        if max_weight < min_weight:
            raise ValueError(f"{key_prefix}.max_weight must be >= {key_prefix}.min_weight.")
        return (
            update_interval_seconds,
            min_weight,
            max_weight,
            lp_balance_tolerance,
            lp_ewma_gamma,
        )

    lp_wrr_cfg: Optional[LpWrrControlConfig] = None
    if "lp-wrr" in payload:
        (
            update_interval_seconds,
            min_weight,
            max_weight,
            lp_balance_tolerance,
            lp_ewma_gamma,
        ) = _parse_wrr_control_block(payload.get("lp-wrr"), "lp-wrr")
        lp_wrr_cfg = LpWrrControlConfig(
            update_interval_seconds=update_interval_seconds,
            min_weight=min_weight,
            max_weight=max_weight,
            lp_balance_tolerance=lp_balance_tolerance,
            lp_ewma_gamma=lp_ewma_gamma,
        )

    sp_wrr_cfg: Optional[SpWrrControlConfig] = None
    if "sp-wrr" in payload:
        (
            update_interval_seconds,
            min_weight,
            max_weight,
            lp_balance_tolerance,
            lp_ewma_gamma,
        ) = _parse_wrr_control_block(payload.get("sp-wrr"), "sp-wrr")
        sp_wrr_cfg = SpWrrControlConfig(
            update_interval_seconds=update_interval_seconds,
            min_weight=min_weight,
            max_weight=max_weight,
            lp_balance_tolerance=lp_balance_tolerance,
            lp_ewma_gamma=lp_ewma_gamma,
        )

    if (lp_wrr_cfg is not None) and (sp_wrr_cfg is not None):
        raise ValueError("controller.lp-wrr and controller.sp-wrr cannot be used together.")

    return ControllerConfig(
        latency_tracker=latency_cfg,
        lp_wrr=lp_wrr_cfg,
        sp_wrr=sp_wrr_cfg,
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
    # Allow shorthand like {"name": "fixed_rate", "rate": 0.05}.
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
        self.completion_count = 0
        self.class_load_balancers: Dict[int, "LoadBalancer"] = {}
        self.latency_trackers_by_class: Dict[int, LatencyTrackerWorker] = {}
        self.wrr_control_mode = "fixed" if self.policy == "static-wrr" else ""
        self.latency_tracker_enabled = self.policy in TRACKER_REQUIRED_POLICIES

        if self.latency_tracker_enabled and (self.config.latency_tracker is None):
            raise ValueError(
                f"Policy '{self.policy}' requires controller.latency_tracker config."
            )
        if (
            self.policy not in {"lp-wrr", "sp-wrr"}
            and (self.config.lp_wrr is not None or self.config.sp_wrr is not None)
        ):
            raise ValueError(
                "controller.lp-wrr/controller.sp-wrr is only supported when policy is "
                "'lp-wrr' or 'sp-wrr'."
            )

        if self.policy == "lp-wrr":
            if self.config.lp_wrr is None:
                raise ValueError(
                    "Policy 'lp-wrr' requires controller.lp-wrr in controller config."
                )
            if self.config.sp_wrr is not None:
                raise ValueError(
                    "controller.sp-wrr is not supported when policy='lp-wrr'."
                )
            self.wrr_control_mode = "lp-wrr"
            self.lb_control_module = create_load_balancer_control_module(
                "wrr_lp_latency",
                num_workers=self.num_workers,
                params=WrrLpControlParams(
                    update_interval_seconds=self.config.lp_wrr.update_interval_seconds,
                    min_weight=self.config.lp_wrr.min_weight,
                    max_weight=self.config.lp_wrr.max_weight,
                    lp_balance_tolerance=self.config.lp_wrr.lp_balance_tolerance,
                    lp_ewma_gamma=self.config.lp_wrr.lp_ewma_gamma,
                ),
            )
        elif self.policy == "sp-wrr":
            if self.config.sp_wrr is None:
                raise ValueError(
                    "Policy 'sp-wrr' requires controller.sp-wrr in controller config."
                )
            if self.config.lp_wrr is not None:
                raise ValueError(
                    "controller.lp-wrr is not supported when policy='sp-wrr'."
                )
            self.wrr_control_mode = "sp-wrr"
            self.lb_control_module = create_load_balancer_control_module(
                "wrr_separate_lp_latency",
                num_workers=self.num_workers,
                params=WrrLpControlParams(
                    update_interval_seconds=self.config.sp_wrr.update_interval_seconds,
                    min_weight=self.config.sp_wrr.min_weight,
                    max_weight=self.config.sp_wrr.max_weight,
                    lp_balance_tolerance=self.config.sp_wrr.lp_balance_tolerance,
                    lp_ewma_gamma=self.config.sp_wrr.lp_ewma_gamma,
                ),
            )
        else:
            if self.policy not in WRR_POLICIES:
                self.wrr_control_mode = ""
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
                if self.config.latency_tracker is None:
                    raise RuntimeError("Missing latency tracker config during controller setup.")
                class_tracker = LatencyTrackerWorker(
                    num_workers=self.num_workers,
                    tracker_worker_id=self.tracker_worker_id,
                    config=self.config.latency_tracker,
                    rng=random.Random(self.rng.randrange(1, 2**31)),
                    allowed_worker_ids=lb.worker_ids,
                )
                self.latency_trackers_by_class[class_id] = class_tracker
                lb.configure_latency_tracker(
                    tracker_worker_id=class_tracker.tracker_worker_id,
                    should_redirect=class_tracker.should_redirect,
                )
                for worker_id, estimate in enumerate(class_tracker.estimates):
                    lb.set_latency_estimate(worker_id, estimate, feedback_count=0)

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
        summary: Dict[str, object] = {
            "policy": self.policy,
            "completions_seen": self.completion_count,
            "lb_count": len(sorted_lb_items),
            "wrr_control_mode": self.wrr_control_mode,
            "lb_control_module": self.lb_control_module.name,
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

"""Controller module for LB parameter control and sampled latency tracking."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scipy.optimize import linprog

from .latency_redirect_policies import create_latency_redirect_policy
from .models import Request

LATENCY_AWARE_POLICIES = frozenset({"peak_ewma", "latency_only"})
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
    update_every_samples: int = 20
    min_weight: float = 0.1
    max_weight: float = 10.0
    lp_balance_tolerance: float = 0.25
    lp_ewma_gamma: float = 0.10
    lp_weight_ema_decay: float = 0.0
    lp_use_tracked_only: bool = False


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


def _to_bool(raw: object, key: str, default: bool) -> bool:
    value = _to_bool_optional(raw, key, None)
    return default if value is None else value


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
        lp_weight_ema_decay=max(
            0.0,
            min(
                0.999999,
                _to_float(
                    wrr_cfg_dict.get("lp_weight_ema_decay"),
                    "wrr.lp_weight_ema_decay",
                    0.0,
                ),
            ),
        ),
        lp_use_tracked_only=_to_bool(
            wrr_cfg_dict.get("lp_use_tracked_only"),
            "wrr.lp_use_tracked_only",
            False,
        ),
    )

    if wrr_cfg.max_weight < wrr_cfg.min_weight:
        raise ValueError("wrr.max_weight must be >= wrr.min_weight.")
    if wrr_cfg.mode not in {"none", "lp_latency"}:
        raise ValueError("wrr.mode must be one of: none, lp_latency.")
    if wrr_cfg.mode == "lp_latency" and latency_cfg.enabled is not True:
        raise ValueError(
            "wrr.mode='lp_latency' requires latency_tracker.enabled=true in controller config."
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


class LatencyTrackerWorker:
    """
    Special worker used for sampled latency tracking.

    The tracker itself has zero processing time and forwards requests to real workers
    using an internal round-robin dispatcher.
    """

    def __init__(
        self,
        num_workers: int,
        tracker_worker_id: int,
        config: LatencyTrackerConfig,
        rng: random.Random,
    ) -> None:
        self.num_workers = num_workers
        self.tracker_worker_id = tracker_worker_id
        self.ewma_gamma = max(0.0, min(1.0, float(config.ewma_gamma)))
        self.estimates: List[float] = [config.init_estimate for _ in range(num_workers)]
        self.sample_counts: List[int] = [0 for _ in range(num_workers)]
        self.sampled_requests = 0
        self.redirect_policy_name = config.redirect_policy.name
        self.redirect_policy_params: Dict[str, object] = dict(config.redirect_policy.params)
        self.redirect_policy = create_latency_redirect_policy(
            config.redirect_policy.name,
            params=config.redirect_policy.params,
        )
        self.forward_mode = str(
            getattr(self.redirect_policy, "forward_mode", "round_robin")
        ).strip()
        if self.forward_mode not in {"round_robin", "selected_worker"}:
            raise ValueError(
                "latency_tracker.redirect_policy has unsupported forward_mode: "
                f"{self.forward_mode}"
            )
        self.redirect_rate = float(getattr(self.redirect_policy, "rate", 0.0))
        self.redirect_policy_params["rate"] = self.redirect_rate
        self.redirect_policy_params["forward_mode"] = self.forward_mode
        self.rng = rng
        self.redirect_decisions = 0
        self.redirected_requests = 0
        self._next_forward_worker = 0
        logger.info(
            "LatencyTrackerWorker initialized worker_id=%d redirect_policy=%s",
            self.tracker_worker_id,
            self.redirect_policy_name,
        )
        logger.debug(
            "LatencyTrackerWorker params rate=%s forward_mode=%s ewma_gamma=%.4f",
            self.redirect_rate,
            self.forward_mode,
            self.ewma_gamma,
        )

    def should_redirect(self, request: Request) -> bool:
        self.redirect_decisions += 1
        selected = self.redirect_policy.should_redirect(request, self.rng)
        if selected:
            self.redirected_requests += 1
        logger.debug(
            "Redirect decision rid=%d selected=%s",
            request.rid,
            selected,
        )
        return selected

    def pick_forward_worker(
        self,
        request: Request,
        selected_worker_id: Optional[int] = None,
    ) -> int:
        del request
        if self.forward_mode == "selected_worker":
            if selected_worker_id is None:
                raise ValueError(
                    "selected_worker_id is required for selected_worker forward mode."
                )
            logger.debug(
                "Forward mode selected_worker -> worker=%d",
                selected_worker_id,
            )
            return selected_worker_id
        worker_id = self._next_forward_worker
        self._next_forward_worker = (self._next_forward_worker + 1) % self.num_workers
        logger.debug("Forward mode round_robin -> worker=%d", worker_id)
        return worker_id

    def observe(self, worker_id: int, latency: float) -> None:
        previous = self.estimates[worker_id]
        gamma = self.ewma_gamma
        estimate = (1.0 - gamma) * previous + gamma * latency
        self.estimates[worker_id] = max(1e-9, estimate)
        self.sample_counts[worker_id] += 1
        self.sampled_requests += 1
        logger.debug(
            "Latency observed worker=%d latency=%.4f estimate=%.4f samples=%d",
            worker_id,
            latency,
            self.estimates[worker_id],
            self.sample_counts[worker_id],
        )


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
        self._wrr_lp_class_latency_estimates: Dict[int, List[float]] = {}
        self._wrr_lp_class_latency_samples: Dict[int, List[int]] = {}
        self._wrr_lp_class_completions_window: Dict[int, int] = defaultdict(int)
        self._wrr_lp_updates = 0
        self._wrr_lp_last_weights: Optional[List[float]] = None
        self._wrr_lp_latency_sampled_total = 0

        requested = config.latency_tracker.enabled
        self.latency_tracker_enabled = bool(requested)

        if (self.policy in LATENCY_AWARE_POLICIES) and (not self.latency_tracker_enabled):
            raise ValueError(
                f"Policy '{self.policy}' requires latency_tracker.enabled=true in controller config."
            )
        if (
            self.policy == "weighted_round_robin"
            and self.config.wrr.mode == "lp_latency"
            and (not self.latency_tracker_enabled)
        ):
            raise ValueError(
                "Policy 'weighted_round_robin' with wrr.mode='lp_latency' requires "
                "latency_tracker.enabled=true in controller config."
            )

        self.latency_tracker: Optional[LatencyTrackerWorker]
        if self.latency_tracker_enabled:
            self.latency_tracker = LatencyTrackerWorker(
                num_workers=num_workers,
                tracker_worker_id=self.tracker_worker_id,
                config=config.latency_tracker,
                rng=random.Random(self.rng.randrange(1, 2**31)),
            )
        else:
            self.latency_tracker = None
        logger.info(
            "LoadBalancerController initialized policy=%s tracker_enabled=%s",
            self.policy,
            self.latency_tracker_enabled,
        )

    def initialize(self, lb: "LoadBalancer") -> None:
        if self.latency_tracker is not None:
            lb.configure_latency_tracker(
                tracker_worker_id=self.latency_tracker.tracker_worker_id,
                should_redirect=self.latency_tracker.should_redirect,
            )
            for worker_id, estimate in enumerate(self.latency_tracker.estimates):
                lb.set_latency_estimate(worker_id, estimate, feedback_count=0)

        if self.config.wrr.weights is not None:
            lb.set_worker_weights(self.config.wrr.weights)
        logger.info("Controller initialized into load balancer")

    def is_latency_tracker_worker(self, worker_id: int) -> bool:
        return (
            self.latency_tracker is not None
            and worker_id == self.latency_tracker.tracker_worker_id
        )

    def forward_via_latency_tracker(
        self,
        request: Request,
        selected_worker_id: Optional[int] = None,
    ) -> int:
        if self.latency_tracker is None:
            raise RuntimeError("Latency tracker is not enabled.")
        worker_id = self.latency_tracker.pick_forward_worker(
            request,
            selected_worker_id=selected_worker_id,
        )
        logger.debug(
            "Latency tracker forwarded rid=%d -> worker=%d",
            request.rid,
            worker_id,
        )
        return worker_id

    def _maybe_update_wrr_weights(self, lb: "LoadBalancer") -> None:
        if self.policy != "weighted_round_robin":
            return
        if self.config.wrr.mode == "lp_latency":
            self._maybe_update_wrr_weights_lp_latency(lb)
            return

    def _record_wrr_lp_observation(
        self,
        request: Request,
        worker_id: int,
        latency: float,
        latency_tracked: bool,
    ) -> None:
        if self.policy != "weighted_round_robin":
            return
        if self.config.wrr.mode != "lp_latency":
            return

        class_id = int(request.class_id)
        self._wrr_lp_class_completions_window[class_id] += 1

        if self.config.wrr.lp_use_tracked_only and not latency_tracked:
            return

        estimates = self._wrr_lp_class_latency_estimates.get(class_id)
        if estimates is None:
            estimates = [self.config.latency_tracker.init_estimate for _ in range(self.num_workers)]
            self._wrr_lp_class_latency_estimates[class_id] = estimates
        samples = self._wrr_lp_class_latency_samples.get(class_id)
        if samples is None:
            samples = [0 for _ in range(self.num_workers)]
            self._wrr_lp_class_latency_samples[class_id] = samples

        previous = estimates[worker_id]
        gamma = self.config.wrr.lp_ewma_gamma
        estimates[worker_id] = max(1e-9, (1.0 - gamma) * previous + gamma * latency)
        samples[worker_id] += 1
        self._wrr_lp_latency_sampled_total += 1

    def _normalize_wrr_weights(self, worker_loads: Sequence[float]) -> List[float]:
        total = sum(max(0.0, float(value)) for value in worker_loads)
        if total <= 0:
            candidate = [1.0 for _ in range(self.num_workers)]
        else:
            candidate = [
                max(1e-9, (max(0.0, float(value)) / total) * self.num_workers)
                for value in worker_loads
            ]

        clipped = [
            min(max(value, self.config.wrr.min_weight), self.config.wrr.max_weight)
            for value in candidate
        ]
        decay = self.config.wrr.lp_weight_ema_decay
        if (decay > 0.0) and (self._wrr_lp_last_weights is not None):
            clipped = [
                decay * self._wrr_lp_last_weights[idx] + (1.0 - decay) * clipped[idx]
                for idx in range(self.num_workers)
            ]
        return clipped

    def _solve_wrr_lp(
        self,
        demand_by_class: Sequence[Tuple[int, float]],
        cost_by_class: Sequence[Sequence[float]],
    ) -> List[float]:
        service_count = len(demand_by_class)
        worker_count = self.num_workers
        if (service_count <= 0) or (worker_count <= 0):
            return [0.0 for _ in range(worker_count)]

        c: List[float] = []
        for class_idx in range(service_count):
            demand = demand_by_class[class_idx][1]
            for worker_idx in range(worker_count):
                c.append(demand * max(1e-9, float(cost_by_class[class_idx][worker_idx])))

        variable_count = service_count * worker_count
        a_eq = [[0.0 for _ in range(variable_count)] for _ in range(service_count)]
        for class_idx in range(service_count):
            base = class_idx * worker_count
            for worker_idx in range(worker_count):
                a_eq[class_idx][base + worker_idx] = 1.0
        b_eq = [1.0 for _ in range(service_count)]

        total_demand = sum(demand for _, demand in demand_by_class)
        target = total_demand / worker_count
        epsilon = self.config.wrr.lp_balance_tolerance * target
        lower = max(0.0, target - epsilon)
        upper = target + epsilon

        a_ub: List[List[float]] = []
        b_ub: List[float] = []
        for worker_idx in range(worker_count):
            row = [0.0 for _ in range(variable_count)]
            for class_idx in range(service_count):
                row[class_idx * worker_count + worker_idx] = demand_by_class[class_idx][1]
            a_ub.append(row)
            b_ub.append(upper)
            a_ub.append([-value for value in row])
            b_ub.append(-lower)

        result = linprog(
            c=c,
            A_eq=a_eq,
            b_eq=b_eq,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[(0.0, 1.0) for _ in range(variable_count)],
            method="highs",
        )
        if (not result.success) or (result.x is None):
            raise RuntimeError(
                "LP solve failed for WRR lp_latency mode. "
                f"status={result.status}, message={result.message}"
            )

        worker_loads = [0.0 for _ in range(worker_count)]
        for class_idx in range(service_count):
            demand = demand_by_class[class_idx][1]
            for worker_idx in range(worker_count):
                value = float(result.x[class_idx * worker_count + worker_idx])
                worker_loads[worker_idx] += demand * max(0.0, value)
        return worker_loads

    def _maybe_update_wrr_weights_lp_latency(self, lb: "LoadBalancer") -> None:
        if self.completion_count <= 0:
            return
        if self.completion_count % self.config.wrr.update_every_samples != 0:
            return
        if not self._wrr_lp_class_completions_window:
            return
        if self.config.wrr.lp_use_tracked_only and self._wrr_lp_latency_sampled_total <= 0:
            return

        demand_by_class = sorted(
            (
                (class_id, float(count))
                for class_id, count in self._wrr_lp_class_completions_window.items()
                if count > 0
            ),
            key=lambda item: item[0],
        )
        if not demand_by_class:
            return

        cost_by_class: List[List[float]] = []
        for class_id, _ in demand_by_class:
            estimates = self._wrr_lp_class_latency_estimates.get(class_id)
            if estimates is None:
                estimates = [max(1e-9, value) for value in lb.lat_ewma]
            row = [max(1e-9, float(value)) for value in estimates[: self.num_workers]]
            if len(row) < self.num_workers:
                row.extend([max(1e-9, value) for value in lb.lat_ewma[len(row) :]])
            cost_by_class.append(row)

        worker_loads = self._solve_wrr_lp(
            demand_by_class=demand_by_class,
            cost_by_class=cost_by_class,
        )

        weights = self._normalize_wrr_weights(worker_loads)
        lb.set_worker_weights(weights)
        self._wrr_lp_last_weights = list(weights)
        self._wrr_lp_updates += 1
        self._wrr_lp_class_completions_window.clear()
        logger.info(
            "WRR LP-latency weights updated solver=scipy_linprog classes=%d",
            len(demand_by_class),
        )

    def on_request_complete(
        self,
        request: Request,
        worker_id: int,
        latency: float,
        latency_tracked: bool,
        lb: "LoadBalancer",
    ) -> None:
        self.completion_count += 1
        self._record_wrr_lp_observation(
            request=request,
            worker_id=worker_id,
            latency=latency,
            latency_tracked=latency_tracked,
        )
        if latency_tracked and self.latency_tracker is not None:
            self.latency_tracker.observe(worker_id, latency)
            lb.set_latency_estimate(
                worker_id,
                self.latency_tracker.estimates[worker_id],
                self.latency_tracker.sample_counts[worker_id],
            )
        self._maybe_update_wrr_weights(lb)
        logger.debug(
            "Completion processed rid=%d worker=%d tracked=%s latency=%.4f",
            request.rid,
            worker_id,
            latency_tracked,
            latency,
        )

    def summarize(self, lb: "LoadBalancer") -> Dict[str, object]:
        summary: Dict[str, object] = {
            "mode": self.control_mode,
            "policy": self.policy,
            "completions_seen": self.completion_count,
            "wrr_control_mode": self.config.wrr.mode,
            "wrr_weights": [float(value) for value in lb.worker_weights],
            "wrr_lp_solver": "scipy_linprog",
            "wrr_lp_updates": self._wrr_lp_updates,
            "wrr_lp_sampled_observations": self._wrr_lp_latency_sampled_total,
            "latency_tracker_enabled": self.latency_tracker is not None,
        }

        if self.latency_tracker is not None:
            summary["track_decisions"] = self.latency_tracker.redirect_decisions
            summary["track_redirected"] = self.latency_tracker.redirected_requests
            summary["latency_sample_rate"] = float(
                self.latency_tracker.redirect_rate
            )
            summary["latency_tracker_ewma_gamma"] = self.latency_tracker.ewma_gamma
            summary["latency_tracker_worker_id"] = self.latency_tracker.tracker_worker_id
            summary["latency_tracker_dispatches"] = lb.latency_tracker_dispatches
            summary["latency_tracker_inflight"] = lb.latency_tracker_inflight
            summary["latency_redirect_policy"] = {
                "name": self.latency_tracker.redirect_policy_name,
                "params": dict(self.latency_tracker.redirect_policy_params),
            }
            summary["latency_samples_total"] = self.latency_tracker.sampled_requests
            summary["latency_samples_by_worker"] = list(self.latency_tracker.sample_counts)
            summary["latency_estimate_by_worker"] = list(self.latency_tracker.estimates)
        else:
            summary["track_decisions"] = 0
            summary["track_redirected"] = 0
            summary["latency_sample_rate"] = 0.0
            summary["latency_tracker_ewma_gamma"] = 0.0
            summary["latency_tracker_worker_id"] = None
            summary["latency_tracker_dispatches"] = 0
            summary["latency_tracker_inflight"] = 0
            summary["latency_redirect_policy"] = {}
            summary["latency_samples_total"] = 0
            summary["latency_samples_by_worker"] = []
            summary["latency_estimate_by_worker"] = []

        logger.info("Controller summary generated")
        return summary

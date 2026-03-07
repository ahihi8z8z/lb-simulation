"""Pluggable modules that control load-balancer parameters over time."""

from __future__ import annotations

import logging
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, Type

import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WrrLpControlParams:
    update_interval_seconds: float
    min_weight: float
    max_weight: float
    lp_balance_tolerance: float
    lp_ewma_gamma: float
    lp_use_tracked_only: bool
    init_latency_estimate: float


class LoadBalancerControlModule(ABC):
    """Base class for controller-side LB control modules."""

    name: str = ""

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers

    def initialize(self, lbs_by_class: Mapping[int, "LoadBalancer"]) -> None:
        del lbs_by_class

    def on_request_complete(
        self,
        request: "Request",
        worker_id: int,
        latency: float,
        latency_tracked: bool,
    ) -> None:
        del request, worker_id, latency, latency_tracked

    def summarize(self, lbs_by_class: Mapping[int, "LoadBalancer"]) -> Dict[str, object]:
        del lbs_by_class
        return {}


_LB_CONTROL_MODULE_REGISTRY: Dict[str, Type[LoadBalancerControlModule]] = {}


def register_load_balancer_control_module(
    module_cls: Type[LoadBalancerControlModule],
) -> Type[LoadBalancerControlModule]:
    key = module_cls.name.strip().lower()
    if not key:
        raise ValueError("LB control module name must be non-empty.")
    if key in _LB_CONTROL_MODULE_REGISTRY:
        raise ValueError(f"Duplicate LB control module registration: {key}")
    _LB_CONTROL_MODULE_REGISTRY[key] = module_cls
    logger.debug("Registered LB control module: %s", key)
    return module_cls


def create_load_balancer_control_module(
    name: str,
    num_workers: int,
    params: object = None,
) -> LoadBalancerControlModule:
    key = name.strip().lower()
    module_cls = _LB_CONTROL_MODULE_REGISTRY.get(key)
    if module_cls is None:
        supported = ", ".join(sorted(_LB_CONTROL_MODULE_REGISTRY.keys()))
        raise ValueError(f"Unknown LB control module: {name}. Available modules: {supported}")
    logger.info("Creating LB control module: %s", key)
    if params is None:
        return module_cls(num_workers=num_workers)
    return module_cls(num_workers=num_workers, params=params)


@register_load_balancer_control_module
class NoOpLbControlModule(LoadBalancerControlModule):
    name = "none"


@register_load_balancer_control_module
class WrrLpLatencyControlModule(LoadBalancerControlModule):
    name = "wrr_lp_latency"

    def __init__(self, num_workers: int, params: WrrLpControlParams) -> None:
        super().__init__(num_workers=num_workers)
        self.params = params
        self.completion_count = 0
        self.class_latency_estimates: Dict[int, np.ndarray] = {}
        self.class_latency_samples: Dict[int, np.ndarray] = {}
        self.class_completions_window: Dict[int, int] = defaultdict(int)
        self.class_load_balancers: Dict[int, "LoadBalancer"] = {}
        self.lp_updates = 0
        self.latency_sampled_total = 0
        self.last_lp_class_order: List[int] = []
        self.last_lp_weight_matrix: List[List[float]] = []
        self.next_update_time = float(
            np.clip(float(params.update_interval_seconds), a_min=1e-9, a_max=None)
        )

    def initialize(self, lbs_by_class: Mapping[int, "LoadBalancer"]) -> None:
        self.class_load_balancers = {int(class_id): lb for class_id, lb in lbs_by_class.items()}

    def on_request_complete(
        self,
        request: "Request",
        worker_id: int,
        latency: float,
        latency_tracked: bool,
    ) -> None:
        self.completion_count += 1
        completion_time = float(
            np.clip(float(request.t_arrival) + float(latency), a_min=0.0, a_max=None)
        )
        class_id = int(request.class_id)
        self.class_completions_window[class_id] += 1

        if self.params.lp_use_tracked_only and not latency_tracked:
            return

        estimates = self.class_latency_estimates.get(class_id)
        if estimates is None:
            estimates = np.full(self.num_workers, self.params.init_latency_estimate, dtype=float)
            self.class_latency_estimates[class_id] = estimates
        samples = self.class_latency_samples.get(class_id)
        if samples is None:
            samples = np.zeros(self.num_workers, dtype=int)
            self.class_latency_samples[class_id] = samples

        previous = float(estimates[worker_id])
        gamma = self.params.lp_ewma_gamma
        estimates[worker_id] = float(
            np.clip((1.0 - gamma) * previous + gamma * latency, a_min=1e-9, a_max=None)
        )
        samples[worker_id] += 1
        self.latency_sampled_total += 1

        while completion_time + 1e-12 >= self.next_update_time:
            self._maybe_update_weights()
            self.next_update_time += self.params.update_interval_seconds

    def _normalize_weights_row(self, row_weights: Sequence[float]) -> List[float]:
        row = np.clip(np.asarray(row_weights, dtype=float), a_min=0.0, a_max=None)
        total = float(row.sum())
        if total <= 0.0:
            candidate = np.ones(self.num_workers, dtype=float)
        else:
            candidate = np.clip(row, a_min=1e-9, a_max=None)
        clipped = np.clip(candidate, self.params.min_weight, self.params.max_weight)
        return clipped.tolist()

    def _solve_lp(
        self,
        demand_by_class: Sequence[Tuple[int, float]],
        cost_by_class: Sequence[Sequence[float]],
    ) -> np.ndarray:
        service_count = len(demand_by_class)
        worker_count = self.num_workers
        if (service_count <= 0) or (worker_count <= 0):
            return np.zeros((service_count, worker_count), dtype=float)
        demand = np.clip(
            np.asarray([float(raw_demand) for _, raw_demand in demand_by_class], dtype=float),
            a_min=0.0,
            a_max=None,
        )
        total_demand = float(demand.sum())
        if total_demand <= 0:
            return np.zeros((service_count, worker_count), dtype=float)

        cost = np.asarray(cost_by_class, dtype=float)
        if cost.shape != (service_count, worker_count):
            raise ValueError(
                "cost_by_class shape mismatch: "
                f"expected ({service_count}, {worker_count}), got {cost.shape}"
            )
        cost = np.maximum(cost, 1e-9)

        # Minimize system-wide mean latency:
        # sum_c,w P(class=c) * cost[c,w] * x[c,w], where P(c)=demand[c]/total_demand.
        class_prob = demand / total_demand
        c = (class_prob[:, None] * cost).reshape(-1)
        variable_count = service_count * worker_count
        a_eq = np.kron(np.eye(service_count), np.ones((1, worker_count)))
        b_eq = np.ones(service_count, dtype=float)

        target = total_demand / worker_count
        epsilon = self.params.lp_balance_tolerance * target
        lower = float(np.clip(target - epsilon, a_min=0.0, a_max=None))
        upper = target + epsilon

        class_offsets = np.arange(service_count) * worker_count
        worker_ids = np.arange(worker_count)
        selected_indices = class_offsets[None, :] + worker_ids[:, None]
        a_ub_upper = np.zeros((worker_count, variable_count), dtype=float)
        a_ub_upper[worker_ids[:, None], selected_indices] = demand[None, :]
        a_ub = np.vstack((a_ub_upper, -a_ub_upper))
        b_ub = np.concatenate(
            (
                np.full(worker_count, upper, dtype=float),
                np.full(worker_count, -lower, dtype=float),
            )
        )

        result = linprog(
            c=c,
            A_eq=a_eq,
            b_eq=b_eq,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=(0.0, 1.0),
            method="highs",
        )
        if (not result.success) or (result.x is None):
            raise RuntimeError(
                "LP solve failed for WRR lp_latency mode. "
                f"status={result.status}, message={result.message}"
            )

        allocation = np.maximum(result.x.reshape(service_count, worker_count), 0.0)
        return allocation

    def _maybe_update_weights(self) -> None:
        if not self.class_completions_window:
            return
        if not self.class_load_balancers:
            return
        if self.params.lp_use_tracked_only and self.latency_sampled_total <= 0:
            return

        demand_by_class = sorted(
            (
                (class_id, float(count))
                for class_id, count in self.class_completions_window.items()
                if (count > 0) and (class_id in self.class_load_balancers)
            ),
            key=lambda item: item[0],
        )
        if not demand_by_class:
            return

        class_count = len(demand_by_class)
        cost_by_class = np.zeros((class_count, self.num_workers), dtype=float)
        for row_idx, (class_id, _) in enumerate(demand_by_class):
            class_lb = self.class_load_balancers[class_id]
            default_cost_row = np.maximum(
                np.asarray(class_lb.lat_ewma[: self.num_workers], dtype=float),
                1e-9,
            )
            cost_by_class[row_idx, :] = default_cost_row
            estimates = self.class_latency_estimates.get(class_id)
            if estimates is None:
                continue

            row_len = min(int(estimates.size), self.num_workers)
            cost_by_class[row_idx, :row_len] = estimates[:row_len]
        cost_by_class = np.clip(cost_by_class, a_min=1e-9, a_max=None)
        logger.info(
            "cost by class %s",
            np.array2string(cost_by_class, precision=8, suppress_small=False),
        )

        allocation = self._solve_lp(
            demand_by_class=demand_by_class,
            cost_by_class=cost_by_class,
        )
        logger.info(
            "lp allocation by class %s",
            np.array2string(allocation, precision=8, suppress_small=False),
        )
        self.last_lp_class_order = [class_id for class_id, _ in demand_by_class]
        self.last_lp_weight_matrix = []
        for row_idx, class_id in enumerate(self.last_lp_class_order):
            row_weights = self._normalize_weights_row(allocation[row_idx, :])
            self.class_load_balancers[class_id].set_worker_weights(row_weights)
            self.last_lp_weight_matrix.append([float(value) for value in row_weights])
        self.lp_updates += 1
        self.class_completions_window.clear()
        logger.info(
            "WRR LP-latency weights updated via module solver=scipy_linprog lb_count=%d classes=%d",
            len(self.class_load_balancers),
            len(demand_by_class),
        )

    def summarize(self, lbs_by_class: Mapping[int, "LoadBalancer"]) -> Dict[str, object]:
        del lbs_by_class
        return {
            "wrr_lp_solver": "scipy_linprog",
            "wrr_lp_updates": self.lp_updates,
            "wrr_lp_sampled_observations": self.latency_sampled_total,
            "wrr_lp_update_interval_seconds": self.params.update_interval_seconds,
            "wrr_lp_class_order": list(self.last_lp_class_order),
            "wrr_lp_weight_matrix": [list(row) for row in self.last_lp_weight_matrix],
        }


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .load_balancer import LoadBalancer
    from .models import Request

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
        self.class_completions_window: Dict[int, int] = defaultdict(int)
        self.class_load_balancers: Dict[int, "LoadBalancer"] = {}
        self.lp_updates = 0
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
        del worker_id, latency_tracked
        self.completion_count += 1
        completion_time = float(
            np.clip(float(request.t_arrival) + float(latency), a_min=0.0, a_max=None)
        )
        class_id = int(request.class_id)
        self.class_completions_window[class_id] += 1

        while completion_time + 1e-12 >= self.next_update_time:
            self._maybe_update_weights()
            self.next_update_time += self.params.update_interval_seconds

    def _normalize_weights_row(
        self,
        row_weights: Sequence[float],
        allowed_mask: Sequence[bool],
    ) -> List[float]:
        row = np.clip(np.asarray(row_weights, dtype=float), a_min=0.0, a_max=None)
        mask = np.asarray(allowed_mask, dtype=bool)
        if row.shape != (self.num_workers,) or mask.shape != (self.num_workers,):
            raise ValueError(
                "row_weights/allowed_mask shape mismatch: "
                f"expected ({self.num_workers},)."
            )
        if not np.any(mask):
            raise ValueError("allowed_mask must include at least one worker.")

        allowed_values = row[mask]
        total = float(allowed_values.sum())
        if total <= 0.0:
            candidate_allowed = np.ones(int(mask.sum()), dtype=float)
        else:
            candidate_allowed = np.clip(allowed_values, a_min=1e-9, a_max=None)
        clipped_allowed = np.clip(
            candidate_allowed,
            self.params.min_weight,
            self.params.max_weight,
        )

        normalized_row = np.zeros(self.num_workers, dtype=float)
        normalized_row[mask] = clipped_allowed
        return normalized_row.tolist()

    def _solve_lp(
        self,
        demand_by_class: Sequence[Tuple[int, float]],
        cost_by_class: Sequence[Sequence[float]],
        allowed_mask_by_class: Sequence[Sequence[bool]],
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
        allowed_mask = np.asarray(allowed_mask_by_class, dtype=bool)
        if allowed_mask.shape != (service_count, worker_count):
            raise ValueError(
                "allowed_mask_by_class shape mismatch: "
                f"expected ({service_count}, {worker_count}), got {allowed_mask.shape}"
            )
        if np.any(~allowed_mask.any(axis=1)):
            raise ValueError("Each class row in allowed_mask_by_class must allow >= 1 worker.")

        # Minimize system-wide mean latency:
        # sum_c,w P(class=c) * cost[c,w] * x[c,w], where P(c)=demand[c]/total_demand.
        class_prob = demand / total_demand
        c = (class_prob[:, None] * cost).reshape(-1)
        variable_count = service_count * worker_count
        a_eq = np.kron(np.eye(service_count), np.ones((1, worker_count)))
        b_eq = np.ones(service_count, dtype=float)

        target = total_demand / float(worker_count)
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
        bounds = [
            (0.0, 1.0) if allowed_mask[class_idx, worker_idx] else (0.0, 0.0)
            for class_idx in range(service_count)
            for worker_idx in range(worker_count)
        ]
        result = linprog(
            c=c,
            A_eq=a_eq,
            b_eq=b_eq,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )
        if (not result.success) or (result.x is None):
            logger.warning(
                "WRR lp_latency LP infeasible; fallback to uniform masked allocation. "
                "status=%s message=%s",
                result.status,
                result.message,
            )
            allowed_counts = allowed_mask.sum(axis=1, keepdims=True).astype(float)
            safe_denominator = np.maximum(allowed_counts, 1.0)
            uniform_masked = allowed_mask.astype(float) / safe_denominator
            return uniform_masked

        allocation = np.maximum(result.x.reshape(service_count, worker_count), 0.0)
        return allocation

    def _maybe_update_weights(self) -> None:
        if not self.class_completions_window:
            return
        if not self.class_load_balancers:
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
        allowed_mask_by_class = np.zeros((class_count, self.num_workers), dtype=bool)
        for row_idx, (class_id, _) in enumerate(demand_by_class):
            class_lb = self.class_load_balancers[class_id]
            cost_row = np.maximum(
                np.asarray(class_lb.lat_ewma[: self.num_workers], dtype=float),
                1e-9,
            )
            cost_by_class[row_idx, :] = cost_row
            allowed_mask_by_class[row_idx, class_lb.worker_ids] = True
        cost_by_class = np.clip(cost_by_class, a_min=1e-9, a_max=None)
        logger.info(
            "cost by class %s",
            np.array2string(cost_by_class, precision=8, suppress_small=False),
        )
        logger.info(
            "allowed mask by class %s",
            np.array2string(allowed_mask_by_class.astype(int), separator=","),
        )

        allocation = self._solve_lp(
            demand_by_class=demand_by_class,
            cost_by_class=cost_by_class,
            allowed_mask_by_class=allowed_mask_by_class,
        )
        logger.info(
            "lp allocation by class %s",
            np.array2string(allocation, precision=8, suppress_small=False),
        )
        self.last_lp_class_order = [class_id for class_id, _ in demand_by_class]
        self.last_lp_weight_matrix = []
        gamma = float(np.clip(self.params.lp_ewma_gamma, a_min=0.0, a_max=1.0))
        for row_idx, class_id in enumerate(self.last_lp_class_order):
            class_lb = self.class_load_balancers[class_id]
            class_allowed_mask = allowed_mask_by_class[row_idx, :]
            solved_weights = np.asarray(
                self._normalize_weights_row(
                    allocation[row_idx, :],
                    allowed_mask=class_allowed_mask,
                ),
                dtype=float,
            )
            previous_weights = np.asarray(class_lb.worker_weights, dtype=float)
            smooth_weights = ((1.0 - gamma) * previous_weights) + (gamma * solved_weights)
            row_weights = self._normalize_weights_row(
                smooth_weights,
                allowed_mask=class_allowed_mask,
            )
            class_lb.set_worker_weights(row_weights)
            self.last_lp_weight_matrix.append([float(value) for value in row_weights])
        self.lp_updates += 1
        self.class_completions_window.clear()
        logger.info(
            "WRR LP-latency weights updated via module=%s solver=scipy_linprog lb_count=%d classes=%d",
            self.name,
            len(self.class_load_balancers),
            len(demand_by_class),
        )

    def summarize(self, lbs_by_class: Mapping[int, "LoadBalancer"]) -> Dict[str, object]:
        allowed_workers_by_class = {
            str(class_id): list(lb.worker_ids)
            for class_id, lb in sorted(lbs_by_class.items(), key=lambda item: item[0])
        }
        return {
            "wrr_lp_solver": "scipy_linprog",
            "wrr_lp_objective": "system_mean_latency",
            "wrr_lp_updates": self.lp_updates,
            "wrr_lp_weight_ema_gamma": self.params.lp_ewma_gamma,
            "wrr_lp_update_interval_seconds": self.params.update_interval_seconds,
            "wrr_lp_class_order": list(self.last_lp_class_order),
            "wrr_lp_weight_matrix": [list(row) for row in self.last_lp_weight_matrix],
            "wrr_lp_allowed_workers_by_class": allowed_workers_by_class,
        }


@register_load_balancer_control_module
class WrrSeparateLpLatencyControlModule(WrrLpLatencyControlModule):
    """Solve a separate LP for each class to minimize class-local mean latency."""

    name = "wrr_separate_lp_latency"

    def _solve_lp(
        self,
        demand_by_class: Sequence[Tuple[int, float]],
        cost_by_class: Sequence[Sequence[float]],
        allowed_mask_by_class: Sequence[Sequence[bool]],
    ) -> np.ndarray:
        del demand_by_class
        cost = np.asarray(cost_by_class, dtype=float)
        if cost.ndim != 2:
            raise ValueError(
                "cost_by_class must be a 2D matrix for separate LP mode."
            )
        service_count, worker_count = cost.shape
        if worker_count != self.num_workers:
            raise ValueError(
                "cost_by_class worker dimension mismatch: "
                f"expected {self.num_workers}, got {worker_count}"
            )
        if (service_count <= 0) or (worker_count <= 0):
            return np.zeros((service_count, worker_count), dtype=float)

        cost = np.maximum(cost, 1e-9)
        allowed_mask = np.asarray(allowed_mask_by_class, dtype=bool)
        if allowed_mask.shape != (service_count, worker_count):
            raise ValueError(
                "allowed_mask_by_class shape mismatch: "
                f"expected ({service_count}, {worker_count}), got {allowed_mask.shape}"
            )
        if np.any(~allowed_mask.any(axis=1)):
            raise ValueError("Each class row in allowed_mask_by_class must allow >= 1 worker.")
        a_eq = np.ones((1, worker_count), dtype=float)
        b_eq = np.ones(1, dtype=float)
        allocation = np.zeros((service_count, worker_count), dtype=float)

        for class_idx in range(service_count):
            bounds = [
                (0.0, 1.0) if allowed_mask[class_idx, worker_idx] else (0.0, 0.0)
                for worker_idx in range(worker_count)
            ]
            result = linprog(
                c=cost[class_idx, :],
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if (not result.success) or (result.x is None):
                raise RuntimeError(
                    "LP solve failed for WRR separate_lp mode. "
                    f"class_row={class_idx}, status={result.status}, message={result.message}"
                )
            allocation[class_idx, :] = np.maximum(result.x, 0.0)
        return allocation

    def summarize(self, lbs_by_class: Mapping[int, "LoadBalancer"]) -> Dict[str, object]:
        summary = super().summarize(lbs_by_class)
        summary["wrr_lp_objective"] = "per_class_mean_latency"
        return summary


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .load_balancer import LoadBalancer
    from .models import Request

"""Pluggable modules that control load-balancer parameters over time."""

from __future__ import annotations

import logging
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Type

from scipy.optimize import linprog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WrrLpControlParams:
    update_interval_seconds: float
    min_weight: float
    max_weight: float
    lp_balance_tolerance: float
    lp_ewma_gamma: float
    lp_weight_ema_decay: float
    lp_use_tracked_only: bool
    init_latency_estimate: float


class LoadBalancerControlModule(ABC):
    """Base class for controller-side LB control modules."""

    name: str = ""

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers

    def initialize(self, lb: "LoadBalancer") -> None:
        del lb

    def on_request_complete(
        self,
        request: "Request",
        worker_id: int,
        latency: float,
        latency_tracked: bool,
        lb: "LoadBalancer",
    ) -> None:
        del request, worker_id, latency, latency_tracked, lb

    def summarize(self, lb: "LoadBalancer") -> Dict[str, object]:
        del lb
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
        self.class_latency_estimates: Dict[int, List[float]] = {}
        self.class_latency_samples: Dict[int, List[int]] = {}
        self.class_completions_window: Dict[int, int] = defaultdict(int)
        self.lp_updates = 0
        self.last_weights: List[float] = []
        self.latency_sampled_total = 0
        self.next_update_time = max(1e-9, float(params.update_interval_seconds))

    def on_request_complete(
        self,
        request: "Request",
        worker_id: int,
        latency: float,
        latency_tracked: bool,
        lb: "LoadBalancer",
    ) -> None:
        self.completion_count += 1
        completion_time = max(0.0, float(request.t_arrival) + float(latency))
        class_id = int(request.class_id)
        self.class_completions_window[class_id] += 1

        if self.params.lp_use_tracked_only and not latency_tracked:
            return

        estimates = self.class_latency_estimates.get(class_id)
        if estimates is None:
            estimates = [self.params.init_latency_estimate for _ in range(self.num_workers)]
            self.class_latency_estimates[class_id] = estimates
        samples = self.class_latency_samples.get(class_id)
        if samples is None:
            samples = [0 for _ in range(self.num_workers)]
            self.class_latency_samples[class_id] = samples

        previous = estimates[worker_id]
        gamma = self.params.lp_ewma_gamma
        estimates[worker_id] = max(1e-9, (1.0 - gamma) * previous + gamma * latency)
        samples[worker_id] += 1
        self.latency_sampled_total += 1

        while completion_time + 1e-12 >= self.next_update_time:
            self._maybe_update_weights(lb)
            self.next_update_time += self.params.update_interval_seconds

    def _normalize_weights(self, worker_loads: Sequence[float]) -> List[float]:
        total = sum(max(0.0, float(value)) for value in worker_loads)
        if total <= 0:
            candidate = [1.0 for _ in range(self.num_workers)]
        else:
            candidate = [
                max(1e-9, (max(0.0, float(value)) / total) * self.num_workers)
                for value in worker_loads
            ]

        clipped = [
            min(max(value, self.params.min_weight), self.params.max_weight)
            for value in candidate
        ]
        decay = self.params.lp_weight_ema_decay
        if (decay > 0.0) and self.last_weights:
            clipped = [
                decay * self.last_weights[idx] + (1.0 - decay) * clipped[idx]
                for idx in range(self.num_workers)
            ]
        return clipped

    def _solve_lp(
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
        epsilon = self.params.lp_balance_tolerance * target
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

    def _maybe_update_weights(self, lb: "LoadBalancer") -> None:
        if not self.class_completions_window:
            return
        if self.params.lp_use_tracked_only and self.latency_sampled_total <= 0:
            return

        demand_by_class = sorted(
            (
                (class_id, float(count))
                for class_id, count in self.class_completions_window.items()
                if count > 0
            ),
            key=lambda item: item[0],
        )
        if not demand_by_class:
            return

        cost_by_class: List[List[float]] = []
        for class_id, _ in demand_by_class:
            estimates = self.class_latency_estimates.get(class_id)
            if estimates is None:
                estimates = [max(1e-9, value) for value in lb.lat_ewma]
            row = [max(1e-9, float(value)) for value in estimates[: self.num_workers]]
            if len(row) < self.num_workers:
                row.extend([max(1e-9, value) for value in lb.lat_ewma[len(row) :]])
            cost_by_class.append(row)

        worker_loads = self._solve_lp(
            demand_by_class=demand_by_class,
            cost_by_class=cost_by_class,
        )
        weights = self._normalize_weights(worker_loads)
        lb.set_worker_weights(weights)
        self.last_weights = list(weights)
        self.lp_updates += 1
        self.class_completions_window.clear()
        logger.info(
            "WRR LP-latency weights updated via module solver=scipy_linprog classes=%d",
            len(demand_by_class),
        )

    def summarize(self, lb: "LoadBalancer") -> Dict[str, object]:
        del lb
        return {
            "wrr_lp_solver": "scipy_linprog",
            "wrr_lp_updates": self.lp_updates,
            "wrr_lp_sampled_observations": self.latency_sampled_total,
            "wrr_lp_update_interval_seconds": self.params.update_interval_seconds,
        }


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .load_balancer import LoadBalancer
    from .models import Request

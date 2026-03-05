"""Load balancer state and policy dispatch."""

import random
from typing import List, Optional, Sequence

from .lb_policies import available_policy_names, create_policy
from .models import Request


class LoadBalancer:
    """Dispatch requests with a pluggable policy over shared LB state."""

    def __init__(
        self,
        num_workers: int,
        policy: str = "latency_only",
        ewma_gamma: float = 0.10,
        init_ewma: float = 0.5,
        explore_coef: float = 0.10,
        epsilon: float = 0.03,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.num_workers = num_workers
        self.policy = policy.strip().lower()
        self.ewma_gamma = ewma_gamma
        self.explore_coef = explore_coef
        self.epsilon = epsilon
        self.rng = rng or random.Random()

        self.lat_ewma: List[float] = [init_ewma for _ in range(num_workers)]
        self.inflight: List[int] = [0 for _ in range(num_workers)]
        self.penalty: List[float] = [0.0 for _ in range(num_workers)]
        self.feedback_count: List[int] = [0 for _ in range(num_workers)]
        self.worker_weights: List[float] = [1.0 for _ in range(num_workers)]
        self._policy_impl = create_policy(self.policy)

    def argmin_score(self, scores: Sequence[float]) -> int:
        min_val = min(scores)
        # Random tie-break avoids persistent bias toward small indexes.
        ties = [i for i, value in enumerate(scores) if value == min_val]
        return self.rng.choice(ties)

    def choose_worker(self, request: Request) -> int:
        return self._policy_impl.choose_worker(request, self)

    def on_dispatch(self, worker_id: int) -> None:
        self.inflight[worker_id] += 1

    def on_complete(self, worker_id: int) -> None:
        self.inflight[worker_id] = max(0, self.inflight[worker_id] - 1)

    def set_latency_estimate(self, worker_id: int, estimate: float, feedback_count: int) -> None:
        """Apply controller-provided latency estimate for a worker."""

        self.lat_ewma[worker_id] = max(1e-9, float(estimate))
        self.feedback_count[worker_id] = max(0, int(feedback_count))

    def set_worker_weights(self, weights: Sequence[float]) -> None:
        """Apply controller-provided worker weights for WRR-like policies."""

        if len(weights) != self.num_workers:
            raise ValueError(
                f"weights length {len(weights)} does not match num_workers {self.num_workers}."
            )
        normalized: List[float] = []
        for idx, value in enumerate(weights):
            weight = float(value)
            if weight <= 0:
                raise ValueError(f"weights[{idx}] must be > 0.")
            normalized.append(weight)
        self.worker_weights = normalized


def supported_policies() -> List[str]:
    """Return supported policy names from the policy registry."""

    return available_policy_names()

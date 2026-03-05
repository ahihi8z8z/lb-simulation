"""Latency-feedback load balancer policies."""

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

    def on_complete(self, worker_id: int, latency: float) -> None:
        previous = self.lat_ewma[worker_id]
        gamma = self.ewma_gamma
        self.lat_ewma[worker_id] = (1.0 - gamma) * previous + gamma * latency
        self.inflight[worker_id] = max(0, self.inflight[worker_id] - 1)
        self.feedback_count[worker_id] += 1


def supported_policies() -> List[str]:
    """Return supported policy names from the policy registry."""

    return available_policy_names()

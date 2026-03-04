"""Latency-feedback load balancer policies."""

import random
from typing import List, Optional, Sequence

from .models import Request


class LoadBalancer:
    """Dispatch requests using only latency feedback and inflight counts."""

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
        self.policy = policy
        self.ewma_gamma = ewma_gamma
        self.explore_coef = explore_coef
        self.epsilon = epsilon
        self.rng = rng or random.Random()

        self.lat_ewma: List[float] = [init_ewma for _ in range(num_workers)]
        self.inflight: List[int] = [0 for _ in range(num_workers)]
        self.penalty: List[float] = [0.0 for _ in range(num_workers)]
        self.feedback_count: List[int] = [0 for _ in range(num_workers)]
        self._rr_idx = 0

    def _argmin_score(self, scores: Sequence[float]) -> int:
        min_val = min(scores)
        # Random tie-break avoids persistent bias toward small indexes.
        ties = [i for i, value in enumerate(scores) if value == min_val]
        return self.rng.choice(ties)

    def choose_worker(self, _request: Request) -> int:
        if self.policy == "random":
            return self.rng.randrange(self.num_workers)

        if self.policy == "round_robin":
            worker_id = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % self.num_workers
            return worker_id

        if self.policy == "least_inflight":
            return self._argmin_score(self.inflight)

        if self.policy == "peak_ewma":
            scores = [
                self.lat_ewma[i] * (1.0 + self.inflight[i]) + self.penalty[i]
                for i in range(self.num_workers)
            ]
            return self._argmin_score(scores)

        if self.policy == "latency_only":
            # Epsilon-greedy keeps low-cost exploration active.
            if self.rng.random() < self.epsilon:
                return self.rng.randrange(self.num_workers)

            # Optimism bonus makes under-sampled workers more likely to be tried.
            scores = []
            for i in range(self.num_workers):
                base = self.lat_ewma[i] * (1.0 + self.inflight[i]) + self.penalty[i]
                bonus = self.explore_coef / ((self.feedback_count[i] + 1) ** 0.5)
                scores.append(base - bonus)
            return self._argmin_score(scores)

        raise ValueError(f"Unknown policy: {self.policy}")

    def on_dispatch(self, worker_id: int) -> None:
        self.inflight[worker_id] += 1

    def on_complete(self, worker_id: int, latency: float) -> None:
        previous = self.lat_ewma[worker_id]
        gamma = self.ewma_gamma
        self.lat_ewma[worker_id] = (1.0 - gamma) * previous + gamma * latency
        self.inflight[worker_id] = max(0, self.inflight[worker_id] - 1)
        self.feedback_count[worker_id] += 1

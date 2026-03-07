"""Load balancer state and policy dispatch."""

import logging
import random
from typing import Callable, Dict, List, Optional, Sequence

from .lb_policies import available_policy_names, create_policy
from .models import Request

logger = logging.getLogger(__name__)


class LoadBalancer:
    """Dispatch requests with a pluggable policy over shared LB state."""

    def __init__(
        self,
        num_workers: int,
        policy: str = "latency_only",
        init_ewma: float = 0.5,
        explore_coef: float = 0.10,
        epsilon: float = 0.03,
        rng: Optional[random.Random] = None,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0.")
        self.num_workers = num_workers
        self.policy = policy.strip().lower()
        self.explore_coef = explore_coef
        self.epsilon = epsilon
        self.rng = rng or random.Random()

        self.lat_ewma: List[float] = [init_ewma for _ in range(num_workers)]
        self.inflight: List[int] = [0 for _ in range(num_workers)]
        self.penalty: List[float] = [0.0 for _ in range(num_workers)]
        self.feedback_count: List[int] = [0 for _ in range(num_workers)]
        uniform_weight = 1.0 / float(num_workers)
        self.worker_weights: List[float] = [uniform_weight for _ in range(num_workers)]
        self._policy_impl = create_policy(self.policy)
        self.latency_tracker_worker_id: Optional[int] = None
        self.latency_tracker_inflight = 0
        self.latency_tracker_dispatches = 0
        self._should_redirect_to_tracker: Optional[Callable[[Request], bool]] = None
        self._redirect_target_by_rid: Dict[int, int] = {}
        logger.info(
            "LoadBalancer initialized policy=%s workers=%d",
            self.policy,
            self.num_workers,
        )

    def argmin_score(self, scores: Sequence[float]) -> int:
        min_val = min(scores)
        # Random tie-break avoids persistent bias toward small indexes.
        ties = [i for i, value in enumerate(scores) if value == min_val]
        return self.rng.choice(ties)

    def choose_worker(self, request: Request) -> int:
        worker_id = self._policy_impl.choose_worker(request, self)
        if self._should_redirect_to_tracker is not None:
            should_redirect = self._should_redirect_to_tracker(request)
            if should_redirect and self.latency_tracker_worker_id is not None:
                self._redirect_target_by_rid[request.rid] = worker_id
                logger.debug(
                    "Request rid=%d redirected to latency tracker (selected_worker=%d)",
                    request.rid,
                    worker_id,
                )
                return self.latency_tracker_worker_id
        logger.debug("Request rid=%d selected worker=%d", request.rid, worker_id)
        return worker_id

    def on_dispatch(self, worker_id: int) -> None:
        if (
            self.latency_tracker_worker_id is not None
            and worker_id == self.latency_tracker_worker_id
        ):
            self.latency_tracker_inflight += 1
            self.latency_tracker_dispatches += 1
            logger.debug(
                "Tracker dispatch worker=%d inflight=%d total=%d",
                worker_id,
                self.latency_tracker_inflight,
                self.latency_tracker_dispatches,
            )
            return
        self.inflight[worker_id] += 1
        logger.debug("Worker dispatch worker=%d inflight=%d", worker_id, self.inflight[worker_id])

    def on_complete(self, worker_id: int) -> None:
        if (
            self.latency_tracker_worker_id is not None
            and worker_id == self.latency_tracker_worker_id
        ):
            self.latency_tracker_inflight = max(0, self.latency_tracker_inflight - 1)
            logger.debug(
                "Tracker complete worker=%d inflight=%d",
                worker_id,
                self.latency_tracker_inflight,
            )
            return
        self.inflight[worker_id] = max(0, self.inflight[worker_id] - 1)
        logger.debug("Worker complete worker=%d inflight=%d", worker_id, self.inflight[worker_id])

    def configure_latency_tracker(
        self,
        tracker_worker_id: int,
        should_redirect: Callable[[Request], bool],
    ) -> None:
        """Configure optional latency-tracker worker redirection."""

        if tracker_worker_id < 0:
            raise ValueError("tracker_worker_id must be >= 0.")
        if tracker_worker_id < self.num_workers:
            raise ValueError(
                "tracker_worker_id must not overlap with real worker indexes."
            )
        self.latency_tracker_worker_id = tracker_worker_id
        self._should_redirect_to_tracker = should_redirect
        logger.info("Latency tracker configured with worker_id=%d", tracker_worker_id)

    def consume_redirect_target(self, request_id: int) -> Optional[int]:
        """Get and clear real worker selected before tracker redirection."""

        selected = self._redirect_target_by_rid.pop(request_id, None)
        logger.debug(
            "Consume redirect target rid=%d selected_worker=%s",
            request_id,
            selected,
        )
        return selected

    def set_latency_estimate(self, worker_id: int, estimate: float, feedback_count: int) -> None:
        """Apply controller-provided latency estimate for a worker."""

        self.lat_ewma[worker_id] = max(1e-9, float(estimate))
        self.feedback_count[worker_id] = max(0, int(feedback_count))
        logger.debug(
            "Updated latency estimate worker=%d estimate=%.6f feedback=%d",
            worker_id,
            self.lat_ewma[worker_id],
            self.feedback_count[worker_id],
        )

    def set_worker_weights(self, weights: Sequence[float]) -> None:
        """Apply controller-provided worker weights for WRR-like policies."""

        if len(weights) != self.num_workers:
            raise ValueError(
                f"weights length {len(weights)} does not match num_workers {self.num_workers}."
            )
        raw_weights: List[float] = []
        for idx, value in enumerate(weights):
            weight = float(value)
            if weight <= 0:
                raise ValueError(f"weights[{idx}] must be > 0.")
            raw_weights.append(weight)

        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("sum(weights) must be > 0.")

        normalized = [value / total for value in raw_weights]
        self.worker_weights = normalized
        logger.info(
            "Updated worker weights for main load balancer "
            "(real_workers=%d tracker_worker_id=%s sum=1): %s",
            self.num_workers,
            self.latency_tracker_worker_id,
            [round(value, 6) for value in normalized],
        )


def supported_policies() -> List[str]:
    """Return supported policy names from the policy registry."""

    return available_policy_names()

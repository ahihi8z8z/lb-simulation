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
        policy: str = "static-wrr",
        worker_ids: Optional[Sequence[int]] = None,
        init_ewma: float = 0.5,
        explore_coef: float = 0.10,
        epsilon: float = 0.03,
        lb_id: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0.")
        self.num_workers = num_workers
        self.policy = policy.strip().lower()
        self.explore_coef = explore_coef
        self.epsilon = epsilon
        self.lb_id = lb_id or "default"
        self.rng = rng or random.Random()
        if worker_ids is None:
            self.worker_ids: List[int] = list(range(num_workers))
        else:
            if not worker_ids:
                raise ValueError("worker_ids must not be empty when provided.")
            seen = set()
            normalized_worker_ids: List[int] = []
            for idx, raw_worker_id in enumerate(worker_ids):
                worker_id = int(raw_worker_id)
                if worker_id < 0 or worker_id >= num_workers:
                    raise ValueError(
                        f"worker_ids[{idx}]={worker_id} is out of range [0, {num_workers - 1}]."
                    )
                if worker_id in seen:
                    raise ValueError(f"Duplicate worker id in worker_ids: {worker_id}.")
                seen.add(worker_id)
                normalized_worker_ids.append(worker_id)
            self.worker_ids = normalized_worker_ids

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
            "LoadBalancer initialized id=%s policy=%s workers=%d allowed_workers=%s",
            self.lb_id,
            self.policy,
            self.num_workers,
            self.worker_ids,
        )

    def argmin_score(
        self,
        scores: Sequence[float],
        candidates: Optional[Sequence[int]] = None,
    ) -> int:
        candidate_ids = list(candidates) if candidates is not None else self.worker_ids
        if not candidate_ids:
            raise ValueError("argmin_score requires at least one candidate worker.")
        min_val = min(scores[worker_id] for worker_id in candidate_ids)
        # Random tie-break avoids persistent bias toward small indexes.
        ties = [worker_id for worker_id in candidate_ids if scores[worker_id] == min_val]
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
        logger.info(
            "Latency tracker configured lb_id=%s worker_id=%d",
            self.lb_id,
            tracker_worker_id,
        )

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
        allowed_worker_ids = set(self.worker_ids)
        raw_weights: List[float] = []
        for idx, value in enumerate(weights):
            weight = float(value)
            if idx in allowed_worker_ids:
                if weight <= 0:
                    raise ValueError(
                        f"weights[{idx}] must be > 0 for allowed workers."
                    )
            elif weight < 0:
                raise ValueError(
                    f"weights[{idx}] must be >= 0 for disallowed workers."
                )
            raw_weights.append(weight)

        total_allowed = sum(raw_weights[idx] for idx in allowed_worker_ids)
        if total_allowed <= 0:
            raise ValueError("sum(weights) over allowed workers must be > 0.")

        normalized = [0.0 for _ in range(self.num_workers)]
        for worker_id in allowed_worker_ids:
            normalized[worker_id] = raw_weights[worker_id] / total_allowed
        self.worker_weights = normalized
        logger.info(
            "Updated worker weights for main load balancer "
            "(lb_id=%s real_workers=%d tracker_worker_id=%s sum=1): %s",
            self.lb_id,
            self.num_workers,
            self.latency_tracker_worker_id,
            [round(value, 6) for value in normalized],
        )


def supported_policies() -> List[str]:
    """Return supported policy names from the policy registry."""

    return available_policy_names()

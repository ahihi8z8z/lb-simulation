"""Latency tracker module used by controller."""

from __future__ import annotations

import logging
import random
from typing import Dict, Optional

from .latency_redirect_policies import create_latency_redirect_policy
from .models import Request

logger = logging.getLogger(__name__)


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
        config,
        rng: random.Random,
    ) -> None:
        self.num_workers = num_workers
        self.tracker_worker_id = tracker_worker_id
        self.ewma_gamma = max(0.0, min(1.0, float(config.ewma_gamma)))
        self.estimates = [config.init_estimate for _ in range(num_workers)]
        self.sample_counts = [0 for _ in range(num_workers)]
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

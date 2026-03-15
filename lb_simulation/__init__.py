"""Latency-only load balancing simulator package."""

import logging

from .runner import main, run_simulation

__all__ = ["main", "run_simulation"]

logger = logging.getLogger(__name__)
logger.info("lb_simulation package initialized")
logger.debug("lb_simulation package imported")

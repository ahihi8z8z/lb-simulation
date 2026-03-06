#!/usr/bin/env python3
"""CLI entrypoint for the latency-only load balancing simulator."""

import logging

from lb_simulation.runner import main

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Launching simulator.py entrypoint")
    main()

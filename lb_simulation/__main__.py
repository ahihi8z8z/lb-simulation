"""Allow running as: python -m lb_simulation."""

import logging

from .runner import main

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Launching lb_simulation via module entrypoint")
    logger.debug("Delegating to runner.main()")
    main()

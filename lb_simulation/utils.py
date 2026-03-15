"""Small utility helpers."""

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def percentile(values: Sequence[float], p: float) -> float:
    """Compute percentile `p` (0-100) with linear interpolation."""

    logger.info("Computing percentile p=%s over n=%d", p, len(values))
    if not values:
        logger.debug("Percentile requested with empty values p=%s", p)
        return 0.0
    if len(values) == 1:
        logger.debug("Percentile requested with single value p=%s", p)
        return float(values[0])

    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (p / 100.0)
    left = int(pos)
    right = min(left + 1, len(sorted_vals) - 1)
    weight = pos - left
    result = sorted_vals[left] * (1.0 - weight) + sorted_vals[right] * weight
    logger.debug("Percentile computed p=%s result=%.6f", p, result)
    return result

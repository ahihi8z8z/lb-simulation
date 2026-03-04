"""Small utility helpers."""

from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Compute percentile `p` (0-100) with linear interpolation."""

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (p / 100.0)
    left = int(pos)
    right = min(left + 1, len(sorted_vals) - 1)
    weight = pos - left
    return sorted_vals[left] * (1.0 - weight) + sorted_vals[right] * weight

"""Small statistical helpers for pilot-scale results."""

import math


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Return the 95% Wilson score interval of a proportion."""
    if total == 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def mcnemar_exact_p_value(only_first_correct: int, only_second_correct: int) -> float:
    """Return the two-sided exact McNemar p-value from the two discordant counts."""
    discordant = only_first_correct + only_second_correct
    if discordant == 0:
        return 1.0
    smaller = min(only_first_correct, only_second_correct)
    tail: float = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)

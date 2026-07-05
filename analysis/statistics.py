# analysis/statistics.py

from dataclasses import dataclass

from analysis.concentration import confidence_interval


@dataclass(slots=True)
class BoundedStatistic:
    """
    Generic bounded statistic.

    observed : observed events
    total    : total trials
    """

    observed: int
    total: int

    lower: float
    upper: float

    @property
    def probability(self) -> float:
        if self.total == 0:
            return 0.0
        return self.observed / self.total

    @property
    def lower_probability(self) -> float:
        if self.total == 0:
            return 0.0
        return self.lower / self.total

    @property
    def upper_probability(self) -> float:
        if self.total == 0:
            return 0.0
        return self.upper / self.total


def build_statistic(
    observed: int,
    total: int,
    epsilon: float,
) -> BoundedStatistic:

    lower, upper = confidence_interval(
        observed=observed,
        n=total,
        epsilon=epsilon,
    )

    return BoundedStatistic(
        observed=observed,
        total=total,
        lower=lower,
        upper=upper,
    )


def build_statistics(
    observed_counts: dict,
    total_counts: dict,
    epsilon: float,
) -> dict:

    result = {}

    for intensity in observed_counts:

        result[intensity] = build_statistic(
            observed=observed_counts[intensity],
            total=total_counts[intensity],
            epsilon=epsilon,
        )

    return result
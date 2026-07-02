# components/detector.py

import numpy as np

from components.pulse import CoherentPulse
import config


class Detector:
    """
    Single Photon Detector
    """

    def __init__(
        self,
        efficiency: float = config.DETECTOR_EFFICIENCY,
        dark_count: float = config.DARK_COUNT_RATE,
    ):

        self.efficiency = efficiency
        self.dark_count = dark_count

    def click_probability(
        self,
        pulse: CoherentPulse,
    ) -> float:

        mu = abs(pulse.alpha) ** 2

        signal = 1 - np.exp(-self.efficiency * mu)

        total = 1 - (1 - signal) * (1 - self.dark_count)

        return total

    def detect(
        self,
        pulse: CoherentPulse,
    ) -> bool:

        probability = self.click_probability(pulse)

        return np.random.rand() < probability
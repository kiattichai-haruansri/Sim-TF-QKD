from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class CoherentPulse:
    """
    Weak coherent optical pulse.

    alpha = sqrt(mu) * exp(i*phi)
    """

    alpha: complex
    mu: float
    phase: float
    wavelength: float

    @property
    def intensity(self) -> float:
        return abs(self.alpha) ** 2
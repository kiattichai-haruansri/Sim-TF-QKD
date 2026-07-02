# components/laser.py

import numpy as np

from components.pulse import CoherentPulse


class Laser:

    def __init__(
        self,
        mu: float,
        wavelength: float = 1550e-9,
    ):
        self.mu = mu
        self.wavelength = wavelength

    def emit(self):

        # สมมติให้ค่า mu แกว่งไปมา 5% จากค่าที่ตั้งไว้
        actual_mu = np.random.normal(self.mu, self.mu * 0.05)
        actual_mu = max(0.0, actual_mu) # ป้องกันค่าติดลบ
        
        alpha = np.sqrt(actual_mu)

        return CoherentPulse(
            alpha=alpha,
            mu=self.mu,
            phase=0.0,
            wavelength=self.wavelength,
        )
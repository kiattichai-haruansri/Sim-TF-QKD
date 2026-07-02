# components/laser.py

import numpy as np

from components.pulse import CoherentPulse,CoherentPulseBatch


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
    
    #Vectorized
    def emit_batch(self, batch_size: int):

        actual_mu = np.random.normal(
            loc=self.mu,
            scale=self.mu * 0.05,
            size=batch_size
        )
        actual_mu = np.clip(actual_mu, 0.0, None)

        alpha = np.sqrt(actual_mu)

        return CoherentPulseBatch(
            alpha=alpha,
            mu=actual_mu,
            phase=np.zeros(batch_size),
            wavelength=self.wavelength
        )
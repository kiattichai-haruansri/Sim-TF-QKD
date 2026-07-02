# components/fiber.py

import numpy as np

from components.pulse import CoherentPulse


class Fiber:
    """
    Optical fiber channel.

    Parameters
    ----------
    length : float
        Fiber length (km)

    loss_db : float
        Fiber attenuation (dB/km)
    """

    def __init__(
        self,
        length: float,
        loss_db: float = 0.2,
    ):
        self.length = length
        self.loss_db = loss_db

    @property
    def transmission(self) -> float:
        """
        Power transmission η
        """
        return 10 ** (-self.loss_db * self.length / 10)

    def propagate(self, pulse: CoherentPulse) -> CoherentPulse:
        """
        Propagate a coherent pulse through the fiber.
        """

        eta = self.transmission

        pulse.alpha *= np.sqrt(eta)

        pulse.mu = abs(pulse.alpha) ** 2

        # --- เพิ่มความสมจริง: Phase Drift ---
        # สมมติให้ความคลาดเคลื่อนของเฟส แปรผันตามระยะทาง (Standard deviation = 0.002 rad / km)
        noise_std_dev = 0.002 * self.length 
        phase_noise = np.random.normal(0, noise_std_dev)
        
        pulse.phase = (pulse.phase + phase_noise) % (2 * np.pi)
        # อัปเดต complex amplitude ตามเฟสที่เพี้ยนไป
        pulse.alpha = np.sqrt(pulse.mu) * np.exp(1j * pulse.phase)

        return pulse
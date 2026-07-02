# protocol/bob.py

import numpy as np
from components.laser import Laser
from components.phase_modulator import PhaseModulator
from components.intensity_modulator import IntensityModulator
import config

class Bob:
    def __init__(self, laser_mu: float = 1.0):
        self.laser = Laser(mu=laser_mu)
        self.pm = PhaseModulator()
        self.im = IntensityModulator()

    def prepare_pulse(self, intensities: list, encoding_phases: list):
        """
        การทำงานเหมือนกับ Alice ทุกประการ เนื่องจาก TF-QKD เป็นระบบสมมาตร
        """
        pulse = self.laser.emit()
        
        global_phase = np.random.uniform(0, 2 * np.pi)
        pulse = self.pm.shift(pulse, global_phase)
        
        chosen_phase = np.random.choice(encoding_phases)
        pulse = self.pm.shift(pulse, chosen_phase)
        
        chosen_intensity = np.random.choice(intensities, p=config.INTENSITY_PROBABILITIES)
        pulse = self.im.modulate(pulse, chosen_intensity)
        
        total_phase = (global_phase + chosen_phase) % (2 * np.pi)
        record = {
            "intensity": chosen_intensity,
            "encoding_phase": chosen_phase,
            "global_phase": global_phase,
            "total_phase": total_phase
        }
        
        return pulse, record
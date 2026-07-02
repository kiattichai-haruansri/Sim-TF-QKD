import numpy as np

from components.laser import Laser
from components.phase_modulator import PhaseModulator
from components.intensity_modulator import IntensityModulator

import config


class User:

    def __init__(self, laser_mu: float = 1.0):

        self.laser = Laser(mu=laser_mu)
        self.pm = PhaseModulator()
        self.im = IntensityModulator()

    # ---------- Single Pulse ----------
    def prepare_pulse(
        self,
        intensities,
        encoding_phases,
    ):

        pulse = self.laser.emit()

        global_phase = np.random.uniform(0, 2*np.pi)

        pulse = self.pm.shift(
            pulse,
            global_phase,
        )

        chosen_phase = np.random.choice(
            encoding_phases
        )

        pulse = self.pm.shift(
            pulse,
            chosen_phase,
        )

        chosen_intensity = np.random.choice(
            intensities,
            p=config.INTENSITY_PROBABILITIES,
        )

        pulse = self.im.modulate(
            pulse,
            chosen_intensity,
        )

        record = {
            "intensity": chosen_intensity,
            "encoding_phase": chosen_phase,
            "global_phase": global_phase,
            "total_phase": (global_phase + chosen_phase) % (2*np.pi),
        }

        return pulse, record

    # ---------- Batch ----------
    def prepare_batch(
        self,
        batch_size,
        intensities,
        encoding_phases,
    ):

        pulse = self.laser.emit_batch(batch_size)

        global_phase = np.random.uniform(
            0,
            2*np.pi,
            size=batch_size,
        )

        pulse = self.pm.shift_batch(
            pulse,
            global_phase,
        )

        chosen_phase = np.random.choice(
            encoding_phases,
            size=batch_size,
        )

        pulse = self.pm.shift_batch(
            pulse,
            chosen_phase,
        )

        chosen_intensity = np.random.choice(
            intensities,
            size=batch_size,
            p=config.INTENSITY_PROBABILITIES,
        )

        pulse = self.im.modulate_batch(
            pulse,
            chosen_intensity,
        )

        record = {
            "intensity": chosen_intensity,
            "encoding_phase": chosen_phase,
            "global_phase": global_phase,
            "total_phase": (global_phase + chosen_phase) % (2*np.pi),
        }

        return pulse, record
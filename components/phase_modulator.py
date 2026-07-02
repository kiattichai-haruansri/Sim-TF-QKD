import numpy as np

from components.pulse import CoherentPulse


class PhaseModulator:

    def randomize(self, pulse: CoherentPulse):

        phase = np.random.uniform(0, 2*np.pi)

        pulse.phase = phase

        pulse.alpha = np.sqrt(pulse.mu) * np.exp(1j*phase)

        return pulse

    def shift(self, pulse: CoherentPulse, phase: float):

        pulse.phase += phase

        pulse.alpha = np.sqrt(pulse.mu) * np.exp(1j*pulse.phase)

        return pulse
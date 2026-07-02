import numpy as np

from components.pulse import CoherentPulse,CoherentPulseBatch


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
    
    #vectorized
    def randomize_batch(
        self,
        pulse: CoherentPulseBatch,
    ):

        phase = np.random.uniform(
            0,
            2*np.pi,
            size=pulse.size,
        )

        pulse.phase = phase
        pulse.alpha = np.sqrt(pulse.mu) * np.exp(1j*phase)

        return pulse
    
    def shift_batch(self, pulse: CoherentPulseBatch, phase: float):

        pulse.phase += phase

        pulse.alpha = np.sqrt(pulse.mu) * np.exp(1j*pulse.phase)

        return pulse
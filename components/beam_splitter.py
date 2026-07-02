import numpy as np

from components.pulse import CoherentPulse,CoherentPulseBatch


class BeamSplitter:
    """
    50:50 Beam Splitter
    """

    def interfere(
        self,
        pulse_a: CoherentPulse,
        pulse_b: CoherentPulse,
    ):

        alpha0 = (pulse_a.alpha + pulse_b.alpha) / np.sqrt(2)
        alpha1 = (pulse_a.alpha - pulse_b.alpha) / np.sqrt(2)

        out0 = CoherentPulse(
            alpha=alpha0,
            mu=abs(alpha0)**2,
            phase=np.angle(alpha0),
            wavelength=pulse_a.wavelength,
        )

        out1 = CoherentPulse(
            alpha=alpha1,
            mu=abs(alpha1)**2,
            phase=np.angle(alpha1),
            wavelength=pulse_a.wavelength,
        )

        return out0, out1
    
    #Vectorized
    def interfere_batch(
        self,
        pulse_a: CoherentPulseBatch,
        pulse_b: CoherentPulseBatch,
    ):

        alpha0 = (pulse_a.alpha + pulse_b.alpha) / np.sqrt(2)
        alpha1 = (pulse_a.alpha - pulse_b.alpha) / np.sqrt(2)

        out0 = CoherentPulseBatch(
            alpha=alpha0,
            mu=np.abs(alpha0) ** 2,
            phase=np.angle(alpha0),
            wavelength=pulse_a.wavelength,
        )

        out1 = CoherentPulseBatch(
            alpha=alpha1,
            mu=np.abs(alpha1) ** 2,
            phase=np.angle(alpha1),
            wavelength=pulse_a.wavelength,
        )

        return out0, out1
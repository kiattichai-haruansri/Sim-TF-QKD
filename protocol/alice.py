# protocol/alice.py

import numpy as np
from components.laser import Laser
from components.phase_modulator import PhaseModulator
from components.intensity_modulator import IntensityModulator
import config

class Alice:
    def __init__(self, laser_mu: float = 1.0):
        # สร้างอุปกรณ์ (Physical Components)
        self.laser = Laser(mu=laser_mu)
        self.pm = PhaseModulator()
        self.im = IntensityModulator()

    def prepare_pulse(self, intensities: list, encoding_phases: list):
        """
        สร้างและปรับแต่งพัลส์ตามโปรโตคอล TF-QKD
        
        Parameters:
        - intensities: ลิสต์ของค่า mu ที่เป็นไปได้ (เช่น [Signal, Decoy, Vacuum])
        - encoding_phases: ลิสต์ของเฟสที่ใช้เข้ารหัส (เช่น [0, np.pi])
        
        Returns:
        - pulse: วัตถุ CoherentPulse ที่พร้อมส่งผ่าน Fiber
        - record: ดิกชันนารีเก็บข้อมูลสถานะที่สุ่มได้ (เก็บไว้เป็นความลับของ Alice)
        """
        pulse = self.laser.emit()
        
        # 1. Global Phase Randomization (ป้องกันการโจมตีแบบเลียนแบบสถานะ)
        global_phase = np.random.uniform(0, 2 * np.pi)
        pulse = self.pm.shift(pulse, global_phase)
        
        # 2. Phase Encoding (เข้ารหัสข้อมูล)
        chosen_phase = np.random.choice(encoding_phases)
        pulse = self.pm.shift(pulse, chosen_phase)
        
        # 3. Intensity Modulation (Decoy-state method)
        chosen_intensity = np.random.choice(intensities, p=config.INTENSITY_PROBABILITIES)
        pulse = self.im.modulate(pulse, chosen_intensity)
        
        # บันทึกข้อมูลสำหรับการทำ Key Sifting และ Parameter Estimation
        total_phase = (global_phase + chosen_phase) % (2 * np.pi)
        record = {
            "intensity": chosen_intensity,
            "encoding_phase": chosen_phase,
            "global_phase": global_phase,
            "total_phase": total_phase
        }
        
        return pulse, record
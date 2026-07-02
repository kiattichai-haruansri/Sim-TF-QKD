# components/intensity_modulator.py

import numpy as np
from components.pulse import CoherentPulse,CoherentPulseBatch


class IntensityModulator:
    """
    Intensity Modulator (IM) สำหรับปรับเปลี่ยนความเข้มแสง (Mean photon number, mu)
    ของ Coherent Pulse โดยยังคงรักษาเฟส (Phase) เดิมเอาไว้ 
    จำเป็นอย่างยิ่งสำหรับการทำ Decoy-State ในโปรโตคอล TF-QKD
    """

    def __init__(self, insertion_loss_db: float = 0.0, extinction_ratio_db: float = np.inf):
        """
        Parameters
        ----------
        insertion_loss_db : float
            ค่าความสูญเสียภายในตัวอุปกรณ์ (dB) สภาพอุดมคติคือ 0.0
        extinction_ratio_db : float
            อัตราส่วนระหว่างความเข้มแสงสูงสุดและต่ำสุดที่ทำได้ (dB) 
            ใช้สำหรับจำลองกรณีที่สถานะ Vacuum มีแสงเล็ดลอด (Leakage) สภาพอุดมคติคือ np.inf
        """
        self.insertion_loss_db = insertion_loss_db
        self.extinction_ratio_db = extinction_ratio_db

    @property
    def insertion_transmission(self) -> float:
        """ค่าการทะลุผ่านของกำลังแสงเนื่องจาก Insertion Loss"""
        return 10 ** (-self.insertion_loss_db / 10)

    def modulate(self, pulse: CoherentPulse, target_mu: float) -> CoherentPulse:
        """
        ปรับความเข้มแสงของพัลส์ให้มีค่า mean photon number เท่ากับ target_mu 
        โดยคำนวณการเปลี่ยนแปลงของ Complex Amplitude (alpha) และคงเฟสเดิมไว้
        """
        # ประยุกต์ใช้ Insertion Loss ของตัวอุปกรณ์ก่อนเป็นอันดับแรก
        eta_in = self.insertion_transmission
        current_mu = pulse.mu * eta_in
        
        # จัดการกรณีจำลอง Extinction Ratio ที่ไม่สมบูรณ์สำหรับสถานะ Vacuum (target_mu = 0)
        if target_mu == 0.0 and not np.isinf(self.extinction_ratio_db):
            # แสงเล็ดลอดสัดส่วนเท่ากับ 10^(-ER/10) ของความเข้มแสงสูงสุด (สมมติให้อิงจากตัวแปรปัจจุบันชั่วคราว)
            leakage_factor = 10 ** (-self.extinction_ratio_db / 10)
            actual_target = current_mu * leakage_factor if current_mu > 0 else 1e-7
        else:
            actual_target = target_mu

        # ปรับเปลี่ยนค่า alpha โดยคำนึงถึงเฟสเดิมของพัลส์
        if current_mu > 0:
            # คำนวณอัตราส่วนการปรับแรงดัน/แอมพลิจูด (Square root ของอัตราส่วนพลังงาน)
            scale_factor = np.sqrt(actual_target / current_mu)
            pulse.alpha = (pulse.alpha * np.sqrt(eta_in)) * scale_factor
        else:
            # หากพัลส์เริ่มต้นเป็น Vacuum แต่ต้องการกำหนดความเข้มแสงใหม่ ให้สร้างตามเฟสเดิม
            pulse.alpha = np.sqrt(actual_target) * np.exp(1j * pulse.phase)

        # อัปเดตค่า mu และรักษาสภาพความสัมพันธ์พารามิเตอร์ภายในออบเจกต์
        pulse.mu = actual_target
        
        return pulse

    def attenuate(self, pulse: CoherentPulse, attenuation_db: float) -> CoherentPulse:
        """
        ฟังก์ชันเสริม: ลดทอนความเข้มแสงลงตามค่า dB ที่กำหนดโดยตรง (Variable Attenuator Mode)
        """
        transmission = 10 ** (-attenuation_db / 10)
        pulse.alpha *= np.sqrt(transmission)
        pulse.mu = abs(pulse.alpha) ** 2
        return pulse
    
    #Vectorized
    def modulate_batch(
        self,
        pulse: CoherentPulseBatch,
        target_mu: np.ndarray,
    ) -> CoherentPulseBatch:

        eta_in = self.insertion_transmission

        current_mu = pulse.mu * eta_in

        actual_target = target_mu.copy()

        if not np.isinf(self.extinction_ratio_db):

            leakage_factor = 10 ** (-self.extinction_ratio_db / 10)

            vacuum_mask = target_mu == 0.0

            leakage = np.where(
                current_mu > 0,
                current_mu * leakage_factor,
                1e-7,
            )

            actual_target[vacuum_mask] = leakage[vacuum_mask]

        nonzero_mask = current_mu > 0

        pulse.alpha[nonzero_mask] = (
            pulse.alpha[nonzero_mask]
            * np.sqrt(eta_in)
            * np.sqrt(
                actual_target[nonzero_mask]
                / current_mu[nonzero_mask]
            )
        )

        zero_mask = ~nonzero_mask

        pulse.alpha[zero_mask] = (
            np.sqrt(actual_target[zero_mask])
            * np.exp(1j * pulse.phase[zero_mask])
        )

        pulse.mu = actual_target

        return pulse
    
    def attenuate_batch(
        self,
        pulse: CoherentPulseBatch,
        attenuation_db: float,
    ) -> CoherentPulseBatch:

        transmission = 10 ** (-attenuation_db / 10)

        pulse.alpha *= np.sqrt(transmission)

        pulse.mu = np.abs(pulse.alpha) ** 2

        return pulse
# analysis/parameter_estimation.py
# รอทำใหม่
import numpy as np

def estimate_decoy_parameters(
    mu: float, nu: float, 
    gain_mu: float, gain_nu: float, gain_vac: float,
    error_mu: float, error_nu: float
):
    """
    ประเมินพารามิเตอร์ Y_1 (Single-photon Yield) และ e_1 (Single-photon Error Rate)
    โดยใช้วิธี Decoy-State แบบ 3 สถานะ (Signal=mu, Decoy=nu, Vacuum=0)
    
    Parameters:
    - gain (Q): อัตราการเกิดเหตุการณ์ (Valid events / Total pulses)
    - error (E): ค่า QBER ของแต่ละสถานะ
    """
    # 1. Vacuum Yield และ Error (ทางทฤษฎี e0 = 0.5 เพราะเป็น Dark count ล้วนๆ)
    Y_0 = gain_vac
    e_0 = 0.5 
    
    # 2. คำนวณ Lower bound ของ Single-photon Yield (Y_1)
    # สมการ: Y_1 >= (mu / (mu*nu - nu^2)) * (Gain_nu * e^nu - Gain_mu * e^mu * (nu^2 / mu^2) - ... )
    term1 = gain_nu * np.exp(nu)
    term2 = gain_mu * np.exp(mu) * ((nu**2) / (mu**2))
    term3 = ((mu**2 - nu**2) / (mu**2)) * Y_0
    
    Y_1_lower = (mu / (mu * nu - nu**2)) * (term1 - term2 - term3)
    # ป้องกันค่าติดลบที่เกิดจากความผันผวนทางสถิติ (Finite-size effect)
    Y_1_lower = max(0.0, Y_1_lower)
    
    # 3. คำนวณ Upper bound ของ Single-photon Error Rate (e_1)
    if Y_1_lower > 0 and nu > 0:
        e_1_upper = (error_nu * gain_nu * np.exp(nu) - e_0 * Y_0) / (Y_1_lower * nu)
        e_1_upper = max(0.0, min(0.5, e_1_upper)) # e_1 ไม่ควรเกิน 50%
    else:
        e_1_upper = 0.5

    # 4. อัตราการได้รับโฟตอนเดี่ยวรวม (Q_1)
    Q_1 = Y_1_lower * mu * np.exp(-mu)

    return {
        "Y_1": Y_1_lower,
        "e_1": e_1_upper,
        "Q_1": Q_1
    }
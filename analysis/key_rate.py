# analysis/key_rate.py
import numpy as np

def shannon_entropy(x: float) -> float:
    if x <= 0.0 or x >= 1.0: return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def calculate_finite_key_rate(
    M_x: float,           # จำนวนข้อมูลที่ผ่าน sifting ใน X basis
    qber_signal: float,   # QBER ของ X basis
    e_ph_U: float,        # ขอบเขตบนของ Phase-error rate (ได้จาก parameter_estimation)
    epsilon_s: float,     # Failure probability ของความปลอดภัย
    lambda_ec: float      # จำนวนบิตที่ใช้ทำ Error correction
) -> float:
    """
    คำนวณ Secure Key Rate ตามสมการ Finite-key (สมการที่ 4 ของเปเปอร์)
    l = floor[ M_x(1 - h(e_ph^U)) - lambda_ec - log2(2/epsilon_s) ]
    """
    # 1. Privacy amplification term
    privacy_term = M_x * (1 - shannon_entropy(e_ph_U))
    
    # 2. Finite-key penalty terms (สมการ 4)
    # log2(2/epsilon_s) คือค่า Penalty จากการประมาณค่าทางสถิติ
    finite_key_penalty = np.log2(2 / epsilon_s) 
    
    # 3. Secure Key Rate (bits per pulse)
    skr = (privacy_term - lambda_ec - finite_key_penalty) / M_x
    
    return max(0.0, skr)
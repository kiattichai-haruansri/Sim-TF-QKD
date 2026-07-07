# analysis/concentration.py

import numpy as np
import config

def solve_a(M_s, epsilon_a, lambda_n_tilde):
    """
    พารามิเตอร์ 'a' จาก Eq. (32)
    """
    n = float(M_s)  # แปลงเป็น float ป้องกันปัญหาคณิตศาสตร์
    
    num = (3 * ((72 * np.sqrt(n) * lambda_n_tilde * (n - lambda_n_tilde) * np.log(epsilon_a))
            - (16 * (n**(3/2)) * (np.log(epsilon_a)**2))
            + (9 * np.sqrt(2) * (n - 2*lambda_n_tilde) * np.sqrt(-(n**2) * np.log(epsilon_a) * (9 * lambda_n_tilde * (n - lambda_n_tilde) - 2 * n * np.log(epsilon_a))))))
    
    den = (4 * (9*n - 8*np.log(epsilon_a) * (9*lambda_n_tilde*(n-lambda_n_tilde) - 2*n*np.log(epsilon_a))))
    
    return num / den

def solve_b(M_s, epsilon_a, lambda_n_tilde):
    """
    พารามิเตอร์ 'b' จาก Eq. (32)
    """
    n = float(M_s)
    
    # แก้ไข: ใส่ลำดับตัวแปรให้ตรงกับ def solve_a
    a = solve_a(n, epsilon_a, lambda_n_tilde) 
    
    num = np.sqrt(18*(a**2) - (16*a**2 + 24*a*np.sqrt(n) + 9*n)*np.log(epsilon_a))
    
    # แก้ไข: ใส่วงเล็บครอบตัวหารทั้งหมด
    den = (3 * np.sqrt(2 * n)) 
    
    return num / den

def calculate_delta_00(M_s, epsilon_a, M_00_U=config.M_00_U):
    """
    คำนวณ Δ_00 เฉพาะสำหรับคู่โฟตอน (n=0, m=0) 
    ตามเปเปอร์: นำ M^U_00 ไปแทนใน \tilde{\Lambda}_n
    """
    if M_s == 0:
        return 0.0
        
    lambda_n_tilde = M_00_U  # ตามเปเปอร์ให้ใช้ค่า M^U_00 แทนได้เลย
    
    a = solve_a(M_s, epsilon_a, lambda_n_tilde)
    b = solve_b(M_s, epsilon_a, lambda_n_tilde)
    
    # Δ_00 = [b + a((2 M^U_00 / M_s) - 1)] * sqrt(M_s)
    delta_00 = (b + a * (2 * M_00_U / M_s - 1)) * np.sqrt(M_s)
    
    return delta_00

def azuma_delta(M_s: int, epsilon_a: float) -> float:
    """
    คำนวณ Δ (Delta) มาตรฐาน สำหรับโฟตอนคู่ปกติ (ยกเว้น 00)
    Δ = sqrt( M_s/2 * ln(1/ε_a) )
    """
    if M_s <= 0:
        return calculate_delta_00(M_s,epsilon_a)

    return np.sqrt(0.5 * M_s * np.log(1.0 / epsilon_a))

def confidence_interval(observed: int, M_s: int, epsilon: float, is_zero_zero: bool = False, M_00_U: float = config.M_00_U) -> tuple[float, float]:
    """
    คำนวณขอบเขตล่างและบน (Confidence Interval)
    """
    # เลือกว่าจะใช้ Delta_00 หรือ Delta ธรรมดา
    if is_zero_zero:
        delta = calculate_delta_00(M_s, epsilon, M_00_U)
    else:
        delta = azuma_delta(M_s, epsilon)

    lower = max(0.0, observed - delta)
    upper = observed + delta

    return lower, upper

def estimate_lambda_n_tidal(total_pulses, intensities, send_probs, dark_count):
    """
    คำนวณค่า Prior Prediction สำหรับ Vacuum events ในฐาน Z
    """
    # 1. คำนวณโอกาสเกิด Vacuum จากความเข้มแสงแต่ละคู่ (Poisson n=0)
    # P(vac) = sum( p_mu * p_nu * exp(-mu) * exp(-nu) )
    prob_vacuum = 0.0
    for i, mu in enumerate(intensities):
        for j, nu in enumerate(intensities):
            # p_mu * p_nu * exp(-mu) * exp(-nu)
            prob_vacuum += (send_probs[i] * send_probs[j]) * (np.exp(-mu) * np.exp(-nu))
            
    # 2. Yield ทางทฤษฎี (Dark count rate * 2 detectors)
    y_00_theory = 2 * dark_count 
    
    # 3. ค่า Lambda_n_Tidal
    lambda_n_tidal = total_pulses * prob_vacuum * y_00_theory
    
    return lambda_n_tidal
import math
import config
import numpy as np


def poisson_probability(
    mu: float,
    n: int,
) -> float:
    """
    P_n(mu) = exp(-mu) mu^n / n!
    """

    return math.exp(-mu) * mu**n / math.factorial(n)

def poisson_vec(k, mean_array):
    # k เป็นตัวเลข (scalar) ส่วน mean_array เป็น Numpy Array
    # เครื่องหมายทางคณิตศาสตร์ของ Numpy จะทำ Element-wise ให้อัตโนมัติ
    mean_array = np.array(mean_array)
    return (mean_array**k * np.exp(-mean_array)) / math.factorial(k)

def poisson_distribution(
    mu: float,
    cutoff: int,
):
    """
    Return
    ------
    [P0,P1,...,Pcutoff]
    """

    return [
        poisson_probability(mu, n)
        for n in range(cutoff + 1)
    ]

def p_z_nm(n,m):
    mu_set = np.array(config.INTENSITIES)
    p_set = np.array(config.INTENSITY_PROBABILITIES) # แปลงเป็น Array ด้วย

    term_mu = p_set * poisson_vec(n, mu_set)
    term_nu = p_set * poisson_vec(m, mu_set)

    p_z_nm = np.sum(term_mu) * np.sum(term_nu)
    return p_z_nm

def p_x_nm(n,m):
    # ดึงค่า MU_SIGNAL มาใช้โดยตรง (เป็นค่า float เดี่ยวๆ ไม่ใช่ list)
    mu = config.MU_SIGNAL
    
    # คำนวณ Poisson แบบตรงไปตรงมาสำหรับค่า mu เดี่ยวๆ
    p_n = (mu**n * math.exp(-mu)) / math.factorial(n)
    p_m = (mu**m * math.exp(-mu)) / math.factorial(m)
    
    # ผลลัพธ์จะเป็นค่า Scalar เดี่ยวๆ แล้ว
    p_x_nm = p_n * p_m

    return p_x_nm

def calculate_bound(p_X, p_Z, MU, Delta_nm, MZ, Delta, S_cut):
    """
    ฟังก์ชันคำนวณสมการขอบเขต (Bounds calculation)
    
    Parameters:
    p_X (2D array/matrix): ค่า p_{nm|X} ขนาด N x M
    p_Z (2D array/matrix): ค่า p_{nm|Z} ขนาด N x M
    MU (2D array/matrix): ค่า MU_{nm} ขนาด N x M
    Delta_nm (2D array/matrix): ค่า Delta_{nm} ขนาด N x M
    MZ (float): ค่าคงที่ MZ
    Delta (float): ค่าคงที่ Delta
    S_cut (int): ค่า Threshold สำหรับ S_cut
    
    Returns:
    float: ผลลัพธ์จากการคำนวณสมการ
    """
    
    # ดึงขนาดของ matrix (สมมติว่า n เริ่มที่ 0 ถึง N-1, และ m เริ่มที่ 0 ถึง M-1)
    N, M = p_X.shape
    
    term1_sum = 0.0
    term2_sum = 0.0
    
    # วนลูปตาม n และ m
    for n in range(N):
        for m in range(M):
            # คำนวณค่า sqrt(p_{nm|X} * p_{nm|Z}) 
            # (ใส่ max(0, val) กันค่าติดลบจาก error ของ floating point)
            p_sqrt = np.sqrt(max(0, p_X[n, m] * p_Z[n, m]))
            
            if (n + m) <= S_cut:
                # กรณี n + m <= S_cut
                mu_delta_sqrt = np.sqrt(max(0, MU[n, m] + Delta_nm[n, m]))
                term1_sum += p_sqrt * mu_delta_sqrt
            else:
                # กรณี n + m > S_cut
                term2_sum += p_sqrt

    # ประกอบสมการส่วนที่เหลือ
    # sqrt(MZ + Delta) * Sum(...)
    term2_total = np.sqrt(max(0, MZ + Delta)) * term2_sum
    
    # ผลลัพธ์ = ( Term1 + Term2 )^2 + Delta
    result = (term1_sum + term2_total)**2 + Delta
    
    return result
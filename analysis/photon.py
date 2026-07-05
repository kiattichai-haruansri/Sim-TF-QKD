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
    mu_set = config.INTENSITIES  # ค่า μ และ ν ที่เป็นไปได้
    p_set = config.SEND_PROBABILITY   # ความน่าจะเป็นในการเลือกความเข้มแสงนั้นๆ (p_μ, p_ν)

    term_mu = p_set * poisson_vec(n, mu_set)
    term_nu = p_set * poisson_vec(m, mu_set)

    p_z_nm = np.sum(term_mu) * np.sum(term_nu)
    return p_z_nm

def p_x_nm(n,m):
    prob_n = poisson_vec(n, config.INTENSITIES)
    prob_m = poisson_vec(m, config.INTENSITIES)

    p_x_nm = prob_n * prob_m

    return p_x_nm
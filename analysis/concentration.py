# analysis/concentration.py

import numpy as np
import config

def solve_a(n,epsilon_a,lambda_n_Tidal = config.LAMBDA_N_TIDAL):
    return ((3 * ((72 * np.sqrt(n) * lambda_n_Tidal * (n - lambda_n_Tidal) * np.log(epsilon_a))
            - (16 * (n**(3/2)) * (np.log(epsilon_a)**2))
            + (9 * np.sqrt(2) * (n - 2*lambda_n_Tidal) * np.sqrt(-(n**2) * np.log(epsilon_a) * (9 * lambda_n_Tidal * (n - lambda_n_Tidal) - 2 * n * np.log(epsilon_a))))))
            /(4 * (9*n - 8*np.log(epsilon_a) * (9*lambda_n_Tidal*(n-lambda_n_Tidal) - 2*n*np.log(epsilon_a))))
            )

def solve_b(n,epsilon_a,lambda_n_Tidal = config.LAMBDA_N_TIDAL):
    
    a = solve_a(n,lambda_n_Tidal,epsilon_a)
    return (np.sqrt(18*(a**2) - (16*a**2 + 24*a*np.sqrt(n) + 9*n)*np.log(epsilon_a))/
            3*np.sqrt(2*n))

def azuma_delta(
    n: int,
    epsilon_a: float,
) -> float:
    """

    Δ = sqrt( n/2 * ln(1/ε) )

    """
    if n == 0:
        return ((solve_b(n,epsilon_a) + solve_a(n,epsilon_a)*(2*config.M_00_U/n - 1))*np.sqrt(n))

    return np.sqrt(
        0.5 * n * np.log(1 / epsilon_a)
    )

def confidence_interval(
    observed: int,
    n: int,
    epsilon: float,
) -> tuple[float, float]:

    delta = azuma_delta(
        n,
        epsilon,
    )

    lower = max(
        0.0,
        observed - delta,
    )

    upper = observed + delta

    return lower, upper
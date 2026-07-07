# analysis/key_rate.py

from __future__ import annotations
import math
import config


def binary_entropy(x: float) -> float:
    """
    Binary Shannon entropy.

    h(x) = -x log2 x - (1-x) log2(1-x)
    """

    if x <= 0.0 or x >= 1.0:
        return 0.0

    return (
        -x * math.log2(x)
        - (1.0 - x) * math.log2(1.0 - x)
    )


def calculate_finite_key_rate(
    *,
    M_x: int,
    e_ph_U: float,
    lambda_ec: float,
    epsilon_s: float = 1e-10,
    epsilon_pa: float = 1e-10,
):
    """
    Finite-key secret key length (Nature 2020 Eq. 4)

        l =
            M_x (1-h(e_ph))
            - lambda_ec
            - log2(2/eps_s)
            - log2(1/(4 eps_pa²))

    Returns
    -------
    dict

        {
            "secret_key_length": ...,
            "secret_key_rate": ...,
            "privacy_term": ...,
            "finite_penalty": ...,
            "pa_penalty": ...
        }
    """

    if M_x <= 0:

        return {

            "secret_key_length": 0,

            "secret_key_rate": 0.0,

            "privacy_term": 0.0,

            "finite_penalty": 0.0,

            "pa_penalty": 0.0,

        }

    #
    # Privacy amplification
    #

    privacy_term = M_x * (

        1.0 - binary_entropy(e_ph_U)

    )

    #
    # Finite-size penalty
    #

    finite_penalty = math.log2(

        2.0 / epsilon_s

    )

    #
    # Privacy amplification penalty
    #

    pa_penalty = math.log2(

        1.0 / (4.0 * epsilon_pa * epsilon_pa)

    )

    #
    # Secret key length
    #

    key_length = (

        privacy_term
        - lambda_ec
        - finite_penalty
        - pa_penalty

    )

    #key_length = max(

        #0.0,
        #key_length,
        #math.floor(key_length),

    #)

    #
    # Secret key rate
    #

    key_rate = key_length / config.TOTAL_PULSE

    return {

        "secret_key_length": int(key_length),

        "secret_key_rate": key_rate,

        "privacy_term": privacy_term,

        "finite_penalty": finite_penalty,

        "pa_penalty": pa_penalty,

    }
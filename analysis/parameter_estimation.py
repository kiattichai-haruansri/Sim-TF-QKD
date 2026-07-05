# analysis/parameter_estimation.py

from __future__ import annotations

from analysis.statistics import build_statistics
from analysis.decoy_solver import DecoyLPSolver


def estimate_decoy_parameters(
    *,
    intensities: list,
    observed_counts: dict,
    total_counts: dict,
    epsilon: float = 1e-10,
    cutoff: int = 12,
):
    """
    Finite-key Decoy-State parameter estimation for TF-QKD.

    Parameters
    ----------
    intensities
        List of decoy intensities.

    observed_counts
        Number of valid detection events. Keyed by (mu_a, mu_b).

    total_counts
        Number of transmitted pulses. Keyed by (mu_a, mu_b).

    epsilon
        Failure probability.

    cutoff
        Photon-number truncation.

    Returns
    -------
    dict
    {
        "statistics": ...,
        "expectation_bounds": ...,
        "lp_result": ...
    }
    """

    #
    # ---------------------------------------
    # Gain confidence intervals
    # ---------------------------------------
    #
    # build_statistics ตอนนี้ต้องรับและคืนค่าเป็น dict ที่มี Key เป็น (mu_a, mu_b)
    statistics = build_statistics(
        observed_counts=observed_counts,
        total_counts=total_counts,
        epsilon=epsilon,
    )

    #
    # ---------------------------------------
    # Observable bounds
    #
    # Q_lower <= Q <= Q_upper
    # ---------------------------------------
    #

    expectation_bounds = {}

    for mu_a in intensities:
        for mu_b in intensities:
            pair = (mu_a, mu_b)

            # นำเงื่อนไข if mu_a != mu_b: continue ออก
            # เพื่อให้ดึงข้อมูลทั้ง 9 คู่ (สำหรับ 3 intensities) มาใช้งาน

            stat = statistics[pair]

            expectation_bounds[pair] = (
                stat.lower_probability,
                stat.upper_probability,
            )

    #
    # ---------------------------------------
    # LP Solver
    # ---------------------------------------
    #

    solver = DecoyLPSolver(
        intensities=intensities,
        cutoff=cutoff,
    )

    lp_result = solver.solve_all(
        expectation_bounds,
    )

    #
    # ---------------------------------------
    #

    return {
        "statistics": statistics,
        "expectation_bounds": expectation_bounds,
        "lp_result": lp_result,
    }
# analysis/phase_error.py

from __future__ import annotations

def estimate_phase_error(
    *,
    yield_lp_result: dict,
    error_lp_result: dict,
):
    """
    Estimate phase error rate using Decoy-State bounds.
    e_ph = E_single_upper / Y_single_lower
    """

    # 1. ดึงค่าขอบเขตล่างของ Yield โฟตอนเดี่ยว
    Y_single_lower = yield_lp_result[(0, 1)]["lower"]

    # 2. ดึงค่าขอบเขตบนของ Error โฟตอนเดี่ยว
    E_single_upper = error_lp_result[(0, 1)]["upper"]

    # 3. คำนวณ Phase Error
    if Y_single_lower <= 0:
        e_ph = 0.5  # Worst-case scenario
    else:
        e_ph = E_single_upper / Y_single_lower
        
        # Phase error ตามทฤษฎีต้องไม่เกิน 0.5 (50%)
        e_ph = min(0.5, max(0.0, e_ph))

    return {
        "phase_error": e_ph,
        "Y_single": Y_single_lower,
        "E_single": E_single_upper,
    }
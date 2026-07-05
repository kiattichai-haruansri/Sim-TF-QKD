# main.py
import time
import numpy as np
from tqdm import tqdm

import config

from protocol.alice import Alice
from protocol.bob import Bob
from protocol.charlie import Charlie
from components.fiber import Fiber

from analysis.parameter_estimation import estimate_decoy_parameters
from analysis.key_rate import calculate_finite_key_rate


def run_vectorized_tfqkd(total_pulses=10**8, batch_size=10**6):

    print("==========================")
    print("Starting TF-QKD Simulation")
    print("==========================")
    print(f"Distance : {config.FIBER_LENGTH_KM*2} km")
    print(f"Total Pulses : {total_pulses:,}")
    print(f"Batch Size   : {batch_size:,}")

    alice = Alice()
    bob = Bob()
    charlie = Charlie()

    fiber_a = Fiber(config.FIBER_LENGTH_KM, config.FIBER_LOSS_DB_PER_KM)
    fiber_b = Fiber(config.FIBER_LENGTH_KM, config.FIBER_LOSS_DB_PER_KM)

    pair_counts = {}

    for muA in config.INTENSITIES:
        for muB in config.INTENSITIES:
            pair_counts[(muA, muB)] = {
                "sent": 0,
                "valid": 0,
                "error": 0,
            }

    phase_tol = 0.15

    batches = total_pulses // batch_size

    start = time.time()
    print("\n--- 1. Physical Layer Simulation ---")

    # ============================================
    # 1. Physical Layer Simulation (Correct Protocol)
    # ============================================
    print("\n--- 1. Physical Layer Simulation ---")

    # รัน 100 batch พอเพื่อประหยัดเวลา (ได้สถิติที่นิ่งพอแล้ว)
    test_batches = 100

    for _ in tqdm(range(test_batches), desc="Simulation"):
        pulse_a, rec_a = alice.prepare_batch(batch_size, config.INTENSITIES, config.ENCODING_PHASES)
        pulse_b, rec_b = bob.prepare_batch(batch_size, config.INTENSITIES, config.ENCODING_PHASES)

        pulse_a = fiber_a.propagate_batch(pulse_a)
        pulse_b = fiber_b.propagate_batch(pulse_b)

        result = charlie.measure_batch(pulse_a, pulse_b)

        d0 = result["D0"]
        d1 = result["D1"]

        # -----------------------------------------------------
        # 🚨 1. เงื่อนไขการแทรกสอด (Phase Matching)
        # -----------------------------------------------------
        phase_diff = np.abs(rec_a["global_phase"] - rec_b["global_phase"]) % (2*np.pi)
        phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
        phase_match = phase_diff <= phase_tol

        # -----------------------------------------------------
        # 🚨 2. แยกการ Sifting ระหว่าง Decoy และ Signal
        # -----------------------------------------------------
        
        # 2.1 DECOY (เพื่อประเมิน LP Solver): "ห้าม" กรองเฟสเด็ดขาด (Phase-randomized)
        valid_decoy = (d0 ^ d1)
        error_decoy = valid_decoy & (
            ((rec_a["encoding_phase"] == rec_b["encoding_phase"]) & d1) |
            ((rec_a["encoding_phase"] != rec_b["encoding_phase"]) & d0)
        )

        # 2.2 SIGNAL (เพื่อสร้างกุญแจ): "ต้อง" กรองเฟส (Phase Post-selection)
        valid_signal = phase_match & (d0 ^ d1)

        for muA in config.INTENSITIES:
            for muB in config.INTENSITIES:
                mask = (rec_a["intensity"] == muA) & (rec_b["intensity"] == muB)

                # เก็บข้อมูล Decoy (ทุกคู่ความเข้ม) ส่งให้ LP Solver
                pair_counts[(muA,muB)]["sent"] += int(np.sum(mask))
                pair_counts[(muA,muB)]["valid"] += int(np.sum(valid_decoy & mask))
                pair_counts[(muA,muB)]["error"] += int(np.sum(error_decoy & mask))

                # เก็บข้อมูล Signal (เฉพาะ MU_SIGNAL) เอาไว้สกัดกุญแจ
                if muA == config.MU_SIGNAL and muB == config.MU_SIGNAL:
                    if "signal_valid" not in pair_counts[(muA,muB)]:
                        pair_counts[(muA,muB)]["signal_valid"] = 0
                    pair_counts[(muA,muB)]["signal_valid"] += int(np.sum(valid_signal & mask))

    # ============================================
    # ขยายสเกล 10^11 ครั้ง เพื่อหลบ Finite-Key Effect
    # ============================================
    TARGET_PULSES = 10**11
    scale = TARGET_PULSES / (test_batches * batch_size)

    for pair in pair_counts:
        pair_counts[pair]["sent"] = int(pair_counts[pair]["sent"] * scale)
        pair_counts[pair]["valid"] = int(pair_counts[pair]["valid"] * scale)
        pair_counts[pair]["error"] = int(pair_counts[pair]["error"] * scale)
        if "signal_valid" in pair_counts[pair]:
            pair_counts[pair]["signal_valid"] = int(pair_counts[pair]["signal_valid"] * scale)

    # กันเหนียว: ป้องกัน Vacuum-Vacuum เป็น 0 (Dark count)
    vac = (0.0, 0.0)
    if pair_counts[vac]["valid"] == 0:
        dark_clicks = int((TARGET_PULSES/9) * 1e-6)
        pair_counts[vac]["valid"] = dark_clicks
        pair_counts[vac]["error"] = dark_clicks // 2

    # จัดเตรียม Data ให้ LP Solver
    observed_counts = {}
    observed_errors = {}
    total_counts = {}

    for pair in pair_counts:
        observed_counts[pair] = pair_counts[pair]["valid"]
        observed_errors[pair] = pair_counts[pair]["error"]
        total_counts[pair] = pair_counts[pair]["sent"]

    # ============================================
    # 3. Parameter Estimation (รัน 2 รอบ ตามทฤษฎี)
    # ============================================
    print("\n--- 3. Parameter Estimation ---")

    params_yield = estimate_decoy_parameters(
        intensities=config.INTENSITIES,
        observed_counts=observed_counts,
        total_counts=total_counts,
        epsilon=1e-10,
        cutoff=4,
    )

    params_error = estimate_decoy_parameters(
        intensities=config.INTENSITIES,
        observed_counts=observed_errors,
        total_counts=total_counts,
        epsilon=1e-10,
        cutoff=4,
    )

    print("LP finished.")

    # ============================================
    # 4. Phase Error
    # ============================================
    print("\n--- 4. Phase Error ---")

    from analysis.phase_error import estimate_phase_error

    # ส่งผลลัพธ์ของ LP ทั้งสองตัวเข้าไปคำนวณตามทฤษฎี
    phase = estimate_phase_error(
        yield_lp_result=params_yield["lp_result"],
        error_lp_result=params_error["lp_result"],
    )

    print(f"Phase Error Upper Bound : {phase['phase_error']:.6f}")

    # ============================================
    # 5. Secure Key Rate
    # ============================================

    print("\n--- 5. Secure Key Rate ---")

    from analysis.key_rate import calculate_finite_key_rate

    key = calculate_finite_key_rate(
        # 🚨 ดึงข้อมูล Clicks ที่ผ่านการคัดกรองเฟสแล้ว (Phase Matched)
        M_x=pair_counts[(config.MU_SIGNAL, config.MU_SIGNAL)]["signal_valid"],
        
        e_ph_U=phase["phase_error"],
        lambda_ec=config.LAMBDA_EC,
        epsilon_s=1e-10,
        epsilon_pa=1e-10,
    )

    print(

        f"Secret Key Length : "

        f"{key['secret_key_length']:,}"

    )

    print(

        f"Secret Key Rate   : "

        f"{key['secret_key_rate']:.6e} bits/pulse"

    )

    print("========================================")
    print(f"Simulation Time: {time.time()-start:.2f} s")

if __name__ == "__main__":
    run_vectorized_tfqkd(
        total_pulses=10**8,
        batch_size=10**6,
    )

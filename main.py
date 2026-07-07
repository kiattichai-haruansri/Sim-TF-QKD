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

    for _ in tqdm(range(batches), desc="Simulation"):

        pulse_a, rec_a = alice.prepare_batch(
            batch_size,
            config.INTENSITIES,
            config.ENCODING_PHASES,
        )

        pulse_b, rec_b = bob.prepare_batch(
            batch_size,
            config.INTENSITIES,
            config.ENCODING_PHASES,
        )

        pulse_a = fiber_a.propagate_batch(pulse_a)
        pulse_b = fiber_b.propagate_batch(pulse_b)

        result = charlie.measure_batch(
            pulse_a,
            pulse_b,
        )

        d0 = result["D0"]
        d1 = result["D1"]

        same_mu = rec_a["intensity"] == rec_b["intensity"]

        phase_diff = np.abs(
            rec_a["global_phase"] - rec_b["global_phase"]
        ) % (2*np.pi)

        phase_diff = np.minimum(
            phase_diff,
            2*np.pi - phase_diff
        )

        phase_match = phase_diff <= phase_tol

        valid = phase_match & (d0 ^ d1)

        error = valid & (
            ((rec_a["encoding_phase"] == rec_b["encoding_phase"]) & d1)
            |
            ((rec_a["encoding_phase"] != rec_b["encoding_phase"]) & d0)
        )

        for muA in config.INTENSITIES:
            for muB in config.INTENSITIES:

                mask = (
                    (rec_a["intensity"] == muA)
                    &
                    (rec_b["intensity"] == muB)
                )

                pair_counts[(muA,muB)]["sent"] += int(np.sum(mask))
                pair_counts[(muA,muB)]["valid"] += int(np.sum(valid & mask))
                pair_counts[(muA,muB)]["error"] += int(np.sum(error & mask))

    """
    def build(mu):
        total = valid_clicks[mu]
        err = error_clicks[mu]

        return {
            "gain": total / sent_counts[mu] if sent_counts[mu] else 0.0,
            "qber": err / total if total else 0.0,
            "total_events": total,
        }
    """
    
    stats = {}

    for pair,data in pair_counts.items():

        sent = data["sent"]
        valid = data["valid"]
        err = data["error"]

        stats[pair] = {
            "gain": valid/sent if sent else 0,
            "qber": err/valid if valid else 0,
            "events": valid
        }
    """
    sig = build(config.MU_SIGNAL)
    dec = build(config.MU_DECOY)
    vac = build(config.MU_VACUUM)
    """

    labels = {
        config.MU_SIGNAL: "Signal",
        config.MU_DECOY: "Decoy",
        config.MU_VACUUM: "Vacuum",
    }

    print("\n--- 2. Data Sifting & QBER Analysis ---")

    for (muA, muB), s in stats.items():

        nameA = labels.get(muA, str(muA))
        nameB = labels.get(muB, str(muB))

        print(
            f"[{nameA}-{nameB}] "
            f"(μA={muA}, μB={muB}) | "
            f"Gain: {s['gain']:.2e} | "
            f"QBER: {s['qber']*100:.2f}% | "
            f"Clicks: {s['events']:,}"
        )

    # ---------------------------------------------------------
    # ต่อจาก print(f"Simulation Time: {time.time()-start:.2f} s")
    # ---------------------------------------------------------

    print("\n--- 3. Parameter Estimation (Decoy-State) ---")
    
    # ดึงข้อมูล counts ให้อยู่ในรูปแบบที่ parameter_estimation ต้องการ
    observed_counts = {pair: data["valid"] for pair, data in pair_counts.items()}
    total_counts = {pair: data["sent"] for pair, data in pair_counts.items()}
    
    # กำหนด Security & Failure probabilities (Step 1)
    epsilon_c = 1e-10
    epsilon_a = 1e-10
    epsilon_s = 1e-10
    epsilon_pa = 1e-10
    S_cut = 4

    # รัน LP Solver เพื่อหาขอบเขต (Step 4)
    pe_results = estimate_decoy_parameters(
        intensities=config.INTENSITIES,
        observed_counts=observed_counts,
        total_counts=total_counts,
        epsilon=epsilon_c,  # ใช้ eps_c ในการประเมินสถิติเบื้องต้น
        cutoff=S_cut,
    )
    lp_result = pe_results["lp_result"]
    print("✓ LP Solver completed bounds estimation.")

    print("\n--- 4. Finite-Key Rate Calculation ---")

    # กำหนดให้คู่ Signal-Signal เป็นฐานสำหรับการสร้างกุญแจ (M_X)
    pair_signal = (config.MU_SIGNAL, config.MU_SIGNAL)
    M_x = observed_counts[pair_signal]
    E_x = stats[pair_signal]["qber"]
    
    # สมมติว่า M_s คือจำนวน valid events ทั้งหมด (M_X + M_Z)
    M_s = sum(observed_counts.values()) 

    # ---------------------------------------------------------
    # คำนวณ Phase Error Rate (Step 5 & 6)
    # ---------------------------------------------------------
    # หมายเหตุ: ตรงนี้ต้องดึงขอบเขตบน (Upper bounds) MU_nm จาก lp_result
    # ของคุณมาจัดเรียงเป็น Matrix แล้วนำไปเข้าสมการตาม Step 5-6 
    # ผมจะจำลองฟังก์ชันและโครงสร้างให้ดูเป็นตัวอย่าง:
    
    import math
    from analysis.key_rate import binary_entropy
    from analysis.concentration import estimate_lambda_n_tidal

    # Step 5: คำนวณค่า Delta 
    # Δ = sqrt( (1 / 2*M_s) * ln(1 / ε_a) )
    Delta = math.sqrt((1.0 / (2.0 * max(M_s, 1))) * math.log(1.0 / epsilon_a))
    
    # TODO: นำสมการคำนวณ Bound ที่ผมเคยเขียนให้ (calculate_bound) มาประยุกต์ใช้ตรงนี้
    # เพื่อคำนวณ N_U_ph จาก MU_nm ที่ได้จาก lp_result, Delta, p_X, p_Z
    # 
    # จำลองค่า N_U_ph ชั่วคราว (คุณต้องแทนที่ด้วยผลจากการคำนวณจริง)
    # N_U_ph = calculate_bound(...) 
    
    # ---------------------------------------------------------
    # คำนวณ Phase Error Rate ของจริง (Step 5 & 6)
    # ---------------------------------------------------------
    import math
    from analysis.photon import p_x_nm,p_z_nm,calculate_bound

    # สมมติว่า total_pulses คือ 10**9 (ดึงมาจากพารามิเตอร์ที่คุณตั้งไว้ใน main)
    total_pulses = config.TOTAL_PULSE

    # คำนวณค่าคาดหวังของ M_00 ล่วงหน้า (Prior)
    lambda_n_tidal = estimate_lambda_n_tidal(
        total_pulses=total_pulses,
        intensities=config.INTENSITIES,
        send_probs=config.INTENSITY_PROBABILITIES,
        dark_count=config.DARK_COUNT_RATE # ต้องแน่ใจว่าใน config มีตัวแปรนี้นะครับ
    )
    from analysis.concentration import calculate_delta_00,azuma_delta
    # --- 1. คำนวณค่า Delta ---
    # Delta มาตรฐาน สำหรับ (n,m) อื่นๆ
    Delta = azuma_delta(M_s,epsilon_a)
    
    #Delta สำหรับ 00
    Delta_00 = calculate_delta_00(M_s, epsilon_a, lambda_n_tidal)
    
    MZ = M_s 
    term1_sum = 0.0
    term2_sum = 0.0
    S_cut = 4 
    MAX_PHOTONS = 8 

    # --- 2. คำนวณ N^U_ph (Upper Bound ของ Phase Error) ---
    for n in range(MAX_PHOTONS):
        for m in range(MAX_PHOTONS):
            
            P_X = float(p_x_nm(n, m))
            P_Z = float(p_z_nm(n, m))
            p_sqrt = math.sqrt(max(0, P_X / P_Z))
            
            if (n + m) <= S_cut:
                # ดึง Yield ของ Decoy-State
                if (n, m) in lp_result:
                    Y_U_nm = float(lp_result[(n, m)]['upper'])
                else:
                    Y_U_nm = 1.0 
                
                # แปลง Yield เป็น Count
                M_U_nm = total_pulses * P_Z * Y_U_nm
                
                # *** เลือกใช้ Delta_nm ให้ถูกตัว ***
                if n == 0 and m == 0:
                    Delta_nm = Delta_00  # ใช้ค่าที่คำนวณจาก Tidal สำหรับ Vacuum
                else:
                    Delta_nm = Delta     # ใช้ค่ามาตรฐานสำหรับโฟตอนคู่อื่นๆ
                
                term1_sum += p_sqrt * math.sqrt(max(0, M_U_nm + Delta_nm))
            else:
                # เทอมตัดทิ้ง (Truncation)
                term2_sum += p_sqrt

    # --- 3. ประกอบสมการหาจำนวน Phase Error (N_U_ph) ---
    term2_total = math.sqrt(max(0, MZ + Delta)) * term2_sum
    N_U_ph = ((float(config.INTENSITY_PROBABILITIES[0]**2)) / (float(config.INTENSITY_PROBABILITIES[1]**2))) * (term1_sum + term2_total)**2 + Delta
    
    # --- 4. คำนวณสัดส่วน Phase Error (e_ph_U) ---
    e_ph_U = N_U_ph / M_x if M_x > 0 else 1.0
    e_ph_U = min(e_ph_U, 1.0 - 1e-10)
    
    print(f"Sifted Key Length (M_x) : {M_x:,}")
    print(f"QBER (E_x)              : {E_x*100:.2f}%")
    print(f"Phase Error (e_ph_U)    : {e_ph_U*100:.2f}%")

    # ---------------------------------------------------------
    # คำนวณ Key Rate (Step 7)
    # ---------------------------------------------------------
    
    # คำนวณ Error Correction cost (lambda_ec)
    # lambda_ec = f_EC * M_X * h(E_X)
    f_ec = 1.15  # Error correction inefficiency factor
    lambda_ec = f_ec * M_x * binary_entropy(E_x)

    # เรียกใช้ฟังก์ชันคำนวณ Key rate ของคุณ
    key_results = calculate_finite_key_rate(
        M_x=M_x,
        e_ph_U=e_ph_U,
        lambda_ec=lambda_ec,
        epsilon_s=epsilon_s,
        epsilon_pa=epsilon_pa,
    )

    print("========================================")
    print("           FINAL KEY RESULTS            ")
    print("========================================")
    print(f"Secret Key Length : {key_results['secret_key_length']:,} bits")
    print(f"Secret Key Rate   : {key_results['secret_key_rate']:.4e} bits/pulse")
    print(f"Privacy Term      : {key_results['privacy_term']:.2f}")
    print(f"Finite Penalty    : {key_results['finite_penalty']:.2f}")
    print(f"PA Penalty        : {key_results['pa_penalty']:.2f}")
    print("========================================")
    print(f"Simulation Time: {time.time()-start:.2f} s")

if __name__ == "__main__":
    run_vectorized_tfqkd(
        total_pulses=config.TOTAL_PULSE,
        batch_size=10**6,
    )

# main.py

import time
import config
from collections import Counter

from protocol.alice import Alice
from protocol.bob import Bob
from protocol.charlie import Charlie
from components.fiber import Fiber

# นำเข้าโมดูล Analysis ทั้งหมดที่เราเขียนไว้
from analysis.qber import sift_data, calculate_qber
from analysis.parameter_estimation import estimate_decoy_parameters
from analysis.key_rate import calculate_finite_key_rate

def run_full_tfqkd_simulation(num_pulses: int = 500000):
    print("========================================")
    print("  🚀 Starting Full TF-QKD Simulation 🚀  ")
    print("========================================")
    print(f"Distance (Alice to Bob) : {config.FIBER_LENGTH_KM * 2} km")
    print(f"Total Pulses to send    : {num_pulses}")
    print("Running... (This might take a few seconds)\n")

    start_time = time.time()

    # 1. สร้าง Nodes และ Channels
    alice = Alice(laser_mu=1.0)
    bob = Bob(laser_mu=1.0)
    charlie = Charlie(
        detector_efficiency=config.DETECTOR_EFFICIENCY,
        dark_count_rate=config.DARK_COUNT_RATE
    )
    fiber_a = Fiber(length=config.FIBER_LENGTH_KM, loss_db=config.FIBER_LOSS_DB_PER_KM)
    fiber_b = Fiber(length=config.FIBER_LENGTH_KM, loss_db=config.FIBER_LOSS_DB_PER_KM)

    detection_logs = []
    
    # ตัวแปรเก็บว่า Alice และ Bob สุ่มได้ความเข้มแสงตรงกันกี่ครั้ง (ใช้คำนวณ Gain)
    sent_counts = {config.MU_SIGNAL: 0, config.MU_DECOY: 0, config.MU_VACUUM: 0}

    # 2. Simulation Loop (Physical & Protocol Layer)
    for _ in range(num_pulses):
        # State Preparation
        pulse_a, record_a = alice.prepare_pulse(config.INTENSITIES, config.ENCODING_PHASES)
        pulse_b, record_b = bob.prepare_pulse(config.INTENSITIES, config.ENCODING_PHASES)

        # เก็บสถิติรอบที่ Alice และ Bob ส่ง Intensity ตรงกัน
        if record_a["intensity"] == record_b["intensity"]:
            sent_counts[record_a["intensity"]] += 1

        # Channel Propagation
        pulse_a = fiber_a.propagate(pulse_a)
        pulse_b = fiber_b.propagate(pulse_b)

        # Measurement
        result = charlie.measure(pulse_a, pulse_b)

        if result != "NONE":
            detection_logs.append({
                "alice": record_a,
                "bob": record_b,
                "result": result
            })

        #Progress Report
        if _ % 1_000_000 == 0 and _ > 0:
            elapsed = time.time() - start_time
            percent = (_ / num_pulses) * 100
            print(f"รันไปแล้ว {_:,} รอบ ({percent:.4f}%) | เวลาที่ใช้: {elapsed:.2f} วินาที")

    sim_time = time.time() - start_time
    print(f"--- 1. Simulation Completed in {sim_time:.2f}s ---")
    print(f"Total Clicks recorded: {len(detection_logs)}")

    # 3. Post-Processing & Analysis (Analysis Layer)
    print("\n--- 2. Data Sifting & QBER Analysis ---")
    
    # 3.1 Sifting (คัดกรองเฟส)
    PHASE_TOLERANCE = 0.15 
    sifted_logs = sift_data(detection_logs, phase_tolerance=PHASE_TOLERANCE)
    print(f"Valid events after Phase Sifting: {len(sifted_logs)}")

    # 3.2 ดึงสถิติของแต่ละ Decoy State
    stats_sig = calculate_qber(sifted_logs, config.MU_SIGNAL)
    stats_dec = calculate_qber(sifted_logs, config.MU_DECOY)
    stats_vac = calculate_qber(sifted_logs, config.MU_VACUUM)

    # คำนวณ Gain (โอกาสเกิด click ต่อ 1 pulse ที่ส่งไป)
    def calc_gain(stats, intensity):
        sent = sent_counts[intensity]
        return stats["total_events"] / sent if sent > 0 else 0.0

    gain_sig = calc_gain(stats_sig, config.MU_SIGNAL)
    gain_dec = calc_gain(stats_dec, config.MU_DECOY)
    gain_vac = calc_gain(stats_vac, config.MU_VACUUM)

    print(f"\n[Signal (mu={config.MU_SIGNAL})] Gain: {gain_sig:.2e} | QBER: {stats_sig['qber']*100:.2f}%")
    print(f"[Decoy  (nu={config.MU_DECOY})] Gain: {gain_dec:.2e} | QBER: {stats_dec['qber']*100:.2f}%")
    print(f"[Vacuum ( 0={config.MU_VACUUM})] Gain: {gain_vac:.2e} | QBER: {stats_vac['qber']*100:.2f}%")

    print("\n--- 3. Parameter Estimation & Key Rate ---")
    
    # 3.3 ประเมินตัวแปรจากโฟตอนเดี่ยว (Decoy-State Method)
    params = estimate_decoy_parameters(
        mu=config.MU_SIGNAL, 
        nu=config.MU_DECOY,
        gain_mu=gain_sig, 
        gain_nu=gain_dec, 
        gain_vac=gain_vac,
        error_mu=stats_sig["qber"], 
        error_nu=stats_dec["qber"]
    )
    
    print(f"Estimated Single-photon Yield (Y_1): {params['Y_1']:.2e}")
    print(f"Estimated Single-photon Error (e_1): {params['e_1']*100:.2f}%")
    
    # 3.4 คำนวณอัตราการสร้างกุญแจที่ปลอดภัย (Secure Key Rate)
    skr = calculate_finite_key_rate(
        M_x=stats_sig["total_events"],    # จำนวนเหตุการณ์ที่ผ่าน Sifting
        qber_signal=stats_sig["qber"],    # QBER ที่วัดได้
        e_ph_U=params["e_1"],             # Phase-error rate ที่สกัดได้จาก Decoy-state
        epsilon_s=1e-10,                  # Failure probability ตามเปเปอร์ [cite: 178, 245]
        lambda_ec=config.LAMBDA_EC        # บิตที่ใช้แก้ Error
    )
    
    print("========================================")
    if skr > 0:
        print(f"✅ SECURE KEY RATE (Finite-Key): {skr:.6e} bits/pulse")
        total_secure_bits = int(skr * sent_counts[config.MU_SIGNAL])
        print(f"✅ Total Secure Bits Extracted: ~{total_secure_bits} bits")
    else:
        print("❌ SECURE KEY RATE: 0.0 bits/pulse")
        print("❌ Key rate is zero due to high finite-key penalties or high noise.")
    print("========================================")

if __name__ == "__main__":
    # แนะนำให้รันที่ 500,000 หรือ 1,000,000 pulses เพื่อลดความผันผวนทางสถิติ
    run_full_tfqkd_simulation(num_pulses=10_000_000)
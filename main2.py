# main.py

import time
import numpy as np
from tqdm import tqdm
import config

# เราจะเรียกใช้เฉพาะฟังก์ชันทางคณิตศาสตร์จาก Analysis Layer
# (ข้าม sift_data และ calculate_qber แบบเดิมไป เพราะเราจะนับสถิติแบบ Vectorized ที่เร็วกว่า)
from analysis.parameter_estimation import estimate_decoy_parameters
from analysis.key_rate import calculate_finite_key_rate

def run_vectorized_tfqkd(total_pulses: int = 10**10, batch_size: int = 10**7):
    print("========================================")
    print(" 🚀 Starting Vectorized TF-QKD Simulation 🚀")
    print("========================================")
    print(f"Total Pulses to send : {total_pulses:.0e} ({(total_pulses):,})")
    print(f"Batch Size           : {batch_size:.0e} ({(batch_size):,})")
    
    # คำนวณจำนวนรอบที่จะต้องรัน
    num_batches = total_pulses // batch_size
    
    # 1. เตรียมพารามิเตอร์ทางฟิสิกส์
    # Loss ของสาย Fiber ขาเดียว (Alice -> Charlie)
    loss_db = config.FIBER_LENGTH_KM * config.FIBER_LOSS_DB_PER_KM
    # Transmittance รวมประสิทธิภาพของ Detector ด้วย
    eta_sys = (10 ** (-loss_db / 10)) * config.DETECTOR_EFFICIENCY
    
    # ตั้งค่าความน่าจะเป็นของการสุ่ม (ถ้าไม่มีใน config จะใช้ค่า Default นี้)
    prob_sig = getattr(config, 'PROB_SIGNAL', 0.7)
    prob_dec = getattr(config, 'PROB_DECOY', 0.2)
    prob_vac = getattr(config, 'PROB_VACUUM', 0.1)
    intensities = np.array([config.MU_SIGNAL, config.MU_DECOY, config.MU_VACUUM])
    probs = np.array([prob_sig, prob_dec, prob_vac])
    
    phases = np.array(config.ENCODING_PHASES)
    
    # ตัวแปรเก็บสถิติรวม (Aggregated Stats)
    sent_counts = {config.MU_SIGNAL: 0, config.MU_DECOY: 0, config.MU_VACUUM: 0}
    valid_clicks = {config.MU_SIGNAL: 0, config.MU_DECOY: 0, config.MU_VACUUM: 0}
    error_clicks = {config.MU_SIGNAL: 0, config.MU_DECOY: 0, config.MU_VACUUM: 0}
    
    PHASE_TOLERANCE = 0.15
    start_time = time.time()

    print("\n[1/3] Running Physical Layer & Sifting (GPU/CPU Vectorized)...")
    
    # 2. Vectorized Simulation Loop
    for _ in tqdm(range(num_batches), desc="Simulating Batches", unit="batch"):
        
        # --- STATE PREPARATION (Alice & Bob) ---
        mu_a = np.random.choice(intensities, size=batch_size, p=probs)
        mu_b = np.random.choice(intensities, size=batch_size, p=probs)
        
        enc_a = np.random.choice(phases, size=batch_size)
        enc_b = np.random.choice(phases, size=batch_size)
        
        gl_a = np.random.uniform(0, 2 * np.pi, size=batch_size)
        gl_b = np.random.uniform(0, 2 * np.pi, size=batch_size)
        
        # นับจำนวนที่ส่งความเข้มตรงกัน
        same_mu_mask = (mu_a == mu_b)
        
        # --- CHANNEL & INTERFERENCE (Charlie) ---
        # คำนวณคลื่นแม่เหล็กไฟฟ้า E-field (ใช้ Complex number e^(i*phase))
        E_a = np.sqrt(mu_a * eta_sys) * np.exp(1j * (enc_a + gl_a))
        E_b = np.sqrt(mu_b * eta_sys) * np.exp(1j * (enc_b + gl_b))
        
        # Beam Splitter Interference
        E_d0 = (E_a + E_b) / np.sqrt(2)
        E_d1 = (E_a - E_b) / np.sqrt(2)
        
        # คำนวณโอกาสที่ Detector จะคลิก P(click) = 1 - e^(-Intensity) + DarkCount
        p_click_0 = 1 - np.exp(-np.abs(E_d0)**2) + config.DARK_COUNT_RATE
        p_click_1 = 1 - np.exp(-np.abs(E_d1)**2) + config.DARK_COUNT_RATE
        
        # จำลองการวัดผลจริงๆ (Measurement)
        click_0 = np.random.rand(batch_size) < p_click_0
        click_1 = np.random.rand(batch_size) < p_click_1
        
        # --- VECTORIZED SIFTING ---
        # 1. เช็ค Phase Tolerance (Global Phase ตรงกัน)
        phase_diff = np.abs(gl_a - gl_b) % (2 * np.pi)
        min_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
        phase_match_mask = min_diff <= PHASE_TOLERANCE
        
        # 2. เลือกเฉพาะเหตุการณ์ที่สมบูรณ์ (Intensity ตรง, Phase ตรง, คลิกแค่ Detector เดียว)
        valid_event_mask = same_mu_mask & phase_match_mask & (click_0 ^ click_1)
        
        # 3. กำหนดเงื่อนไขการเกิด Error
        # ถ้าเข้ารหัสเหมือนกัน D0 ควรคลิก (ถ้า D1 คลิก = Error)
        # ถ้าเข้ารหัสต่างกัน D1 ควรคลิก (ถ้า D0 คลิก = Error)
        error_mask = valid_event_mask & (((enc_a == enc_b) & click_1) | ((enc_a != enc_b) & click_0))
        
        # --- UPDATE STATISTICS ---
        for mu_val in intensities:
            mu_mask = (mu_a == mu_val)
            
            # จำนวนพัลส์ที่ส่งด้วย Intensity เดียวกัน (และตรงกันทั้งคู่)
            sent_counts[mu_val] += np.sum(same_mu_mask & mu_mask)
            
            # จำนวนที่ผ่าน Sifting
            valid_clicks[mu_val] += np.sum(valid_event_mask & mu_mask)
            
            # จำนวนที่ Error
            error_clicks[mu_val] += np.sum(error_mask & mu_mask)

    sim_time = time.time() - start_time
    print(f"\n✅ Simulation Completed in {sim_time:.2f} seconds!")

    # 3. Data Formatting & Parameter Estimation
    print("\n[2/3] Data Sifting & QBER Analysis")
    
    # สร้าง Stats Dictionary ให้เหมือนที่ฟังก์ชันเก่าคาดหวัง
    def build_stats(mu_val):
        valid = valid_clicks[mu_val]
        errs = error_clicks[mu_val]
        qber = errs / valid if valid > 0 else 0.0
        gain = valid / sent_counts[mu_val] if sent_counts[mu_val] > 0 else 0.0
        return {"total_events": valid, "error_events": errs, "qber": qber, "gain": gain}

    stats_sig = build_stats(config.MU_SIGNAL)
    stats_dec = build_stats(config.MU_DECOY)
    stats_vac = build_stats(config.MU_VACUUM)
    
    print(f"[Signal (mu={config.MU_SIGNAL})] Gain: {stats_sig['gain']:.2e} | QBER: {stats_sig['qber']*100:.2f}% | Clicks: {stats_sig['total_events']:,}")
    print(f"[Decoy  (nu={config.MU_DECOY})] Gain: {stats_dec['gain']:.2e} | QBER: {stats_dec['qber']*100:.2f}% | Clicks: {stats_dec['total_events']:,}")
    print(f"[Vacuum ( 0={config.MU_VACUUM})] Gain: {stats_vac['gain']:.2e} | QBER: {stats_vac['qber']*100:.2f}% | Clicks: {stats_vac['total_events']:,}")

    print("\n[3/3] Parameter Estimation & Key Rate")
    
    # 3.3 ประเมินตัวแปรจากโฟตอนเดี่ยว (Decoy-State Method)
    params = estimate_decoy_parameters(
        mu=config.MU_SIGNAL, 
        nu=config.MU_DECOY,
        gain_mu=stats_sig["gain"], 
        gain_nu=stats_dec["gain"], 
        gain_vac=stats_vac["gain"],
        error_mu=stats_sig["qber"], 
        error_nu=stats_dec["qber"]
    )
    
    print(f"Estimated Single-photon Yield (Y_1): {params['Y_1']:.2e}")
    print(f"Estimated Single-photon Error (e_1): {params['e_1']*100:.2f}%")
    
    # 3.4 คำนวณอัตราการสร้างกุญแจที่ปลอดภัย (Secure Key Rate)
    skr = calculate_finite_key_rate(
        M_x=stats_sig["total_events"],    
        qber_signal=stats_sig["qber"],    
        e_ph_U=params["e_1"],             
        epsilon_s=1e-10,                  
        lambda_ec=config.LAMBDA_EC        
    )
    
    print("\n========================================")
    if skr > 0:
        print(f"✅ SECURE KEY RATE (Finite-Key): {skr:.6e} bits/pulse")
        total_secure_bits = int(skr * sent_counts[config.MU_SIGNAL])
        print(f"✅ Total Secure Bits Extracted: ~{total_secure_bits:,} bits")
    else:
        print("❌ SECURE KEY RATE: 0.0 bits/pulse")
        print("❌ Key rate is zero due to high finite-key penalties or high noise.")
    print("========================================")

if __name__ == "__main__":
    # รัน 10,000,000,000 พัลส์ โดยแบ่งทำทีละ 10,000,000 พัลส์ 
    # (ใช้เวลาไม่กี่นาที แทนที่จะเป็นอาทิตย์!)
    run_vectorized_tfqkd(total_pulses=10**10, batch_size=10**7)
# analysis/qber.py

import numpy as np

def sift_data(detection_logs: list, phase_tolerance: float = 0.1) -> list:
    """
    คัดกรองข้อมูล (Phase Post-selection) 
    เลือกเฉพาะรอบที่ Global phase ของ Alice และ Bob ตรงกันหรือใกล้เคียงกันมาก
    """
    sifted_logs = []
    
    for log in detection_logs:
        # ดึงค่า Global phase
        phi_gA = log["alice"]["global_phase"]
        phi_gB = log["bob"]["global_phase"]
        
        # คำนวณความต่างเฟส (Phase difference) ให้อยู่ในช่วง [-pi, pi]
        delta_phi_g = (phi_gA - phi_gB + np.pi) % (2 * np.pi) - np.pi
        
        # เก็บเฉพาะข้อมูลที่ Phase ตรงกันภายใต้ขอบเขต tolerance
        if abs(delta_phi_g) <= phase_tolerance:
            sifted_logs.append(log)
            
    return sifted_logs

def calculate_qber(sifted_logs: list, target_intensity: float) -> dict:
    """
    คำนวณ Quantum Bit Error Rate (QBER) จากข้อมูลที่ผ่านการ Sift แล้ว
    รองรับการระบุ target_intensity เพื่อแยกคำนวณ QBER ของ Signal หรือ Decoy
    """
    total_valid_events = 0
    error_events = 0
    
    for log in sifted_logs:
        # กรองเฉพาะสถานะความเข้มที่ต้องการ (เช่น ดูเฉพาะ Signal-Signal)
        if log["alice"]["intensity"] != target_intensity or log["bob"]["intensity"] != target_intensity:
            continue
            
        # ข้ามกรณีที่มีการคลิกซ้อน (DOUBLE) หรือไม่คลิก
        if log["result"] not in ["D0", "D1"]:
            continue
            
        phi_eA = log["alice"]["encoding_phase"]
        phi_eB = log["bob"]["encoding_phase"]
        detector = log["result"]
        
        total_valid_events += 1
        
        # ตรวจสอบความถูกต้องของการแทรกสอด
        is_same_bit = np.isclose(phi_eA, phi_eB)
        
        if is_same_bit:
            # บิตตรงกัน: D0 ควรคลิก, ถ้า D1 คลิกคือ Error
            if detector == "D1":
                error_events += 1
        else:
            # บิตต่างกัน: D1 ควรคลิก, ถ้า D0 คลิกคือ Error
            if detector == "D0":
                error_events += 1

    # ป้องกันการหารด้วยศูนย์กรณีที่ไม่มีข้อมูลผ่านเงื่อนไขเลย
    qber = (error_events / total_valid_events) if total_valid_events > 0 else 0.0
    
    return {
        "intensity": target_intensity,
        "total_events": total_valid_events,
        "errors": error_events,
        "qber": qber
    }
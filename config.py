# config.py

import numpy as np

# ==========================================
# Decoy-State Parameters (ความเข้มแสง)
# ==========================================
MU_SIGNAL = 0.5    # ความเข้มแสงสำหรับสถานะ Signal (ข้อมูลหลัก)
MU_DECOY = 0.1     # ความเข้มแสงสำหรับสถานะ Decoy (ตัวล่อตรวจจับการดักฟัง)
MU_VACUUM = 0.0    # ความเข้มแสงสำหรับสถานะ Vacuum
INTENSITIES = [MU_SIGNAL, MU_DECOY, MU_VACUUM]

# ความน่าจะเป็นที่จะสุ่มโดนแต่ละสถานะ
# ตัวอย่าง: ส่ง Signal 70%, Decoy 20%, Vacuum 10%
INTENSITY_PROBABILITIES = [0.7, 0.2, 0.1]

# ==========================================
# Phase Encoding Parameters (การเข้ารหัส)
# ==========================================
# ใน TF-QKD มักใช้เฟส 0 และ pi ในการเข้ารหัสบิต (0 และ 1)
ENCODING_PHASES = [0.0, np.pi]

# ==========================================
# Channel (Fiber) Parameters
# ==========================================
FIBER_LENGTH_KM = 125.0       # ระยะทางไฟเบอร์ฝั่งละ (Alice -> Charlie และ Bob -> Charlie)
FIBER_LOSS_DB_PER_KM = 0.2   # อัตราการลดทอนของสายไฟเบอร์ (dB/km)

# ==========================================
# Detector Parameters (Charlie's Setup)
# ==========================================
DETECTOR_EFFICIENCY = 0.6    # ประสิทธิภาพของ Single Photon Detector (60%)
DARK_COUNT_RATE = 1e-6       # โอกาสเกิด Dark count ต่อพัลส์

# อัปเดตเพิ่มเติมสำหรับ Finite-key
LAMBDA_EC = 0.5   # ปริมาณบิตที่ใช้ทำ Error Correction (หรือคำนวณจาก f * M * h(E))
EPSILON_S = 1e-10 # ค่าความปลอดภัยของระบบ [cite: 245]

# เพิ่มเข้าไปเพื่อให้ SNS ทำงานได้
#SEND_PROBABILITY = 0.5  # ความน่าจะเป็นที่จะส่งสัญญาณ (Sending state)

LAMBDA_N_TIDAL = 1e2
M_00_U = 1e2

TOTAL_PULSE=10**9
# protocol/charlie.py

from components.beam_splitter import BeamSplitter
from components.detector import Detector
import config

class Charlie:
    def __init__(self, detector_efficiency: float = config.DETECTOR_EFFICIENCY, dark_count_rate: float = config.DARK_COUNT_RATE):
        # สร้างอุปกรณ์ (Physical Components)
        self.bs = BeamSplitter()
        self.detector0 = Detector(efficiency=detector_efficiency, dark_count=dark_count_rate)
        self.detector1 = Detector(efficiency=detector_efficiency, dark_count=dark_count_rate)

    def measure(self, pulse_alice, pulse_bob):
        """
        รับพัลส์จาก Alice และ Bob นำมาแทรกสอดกันและวัดผล
        
        Returns:
        - string: รหัสผลลัพธ์การคลิกของ Detector ("D0", "D1", "DOUBLE", "NONE")
        """
        # 1. แทรกสอดสัญญาณผ่าน Beam Splitter
        out0, out1 = self.bs.interfere(pulse_alice, pulse_bob)
        
        # 2. ตรวจจับโฟตอน
        d0_click = self.detector0.detect(out0)
        d1_click = self.detector1.detect(out1)
        
        # 3. ส่งมอบผลลัพธ์ (ในโลกจริงคือการประกาศผ่าน Public Channel)
        if d0_click and d1_click:
            return "DOUBLE"
        elif d0_click:
            return "D0"
        elif d1_click:
            return "D1"
        else:
            return "NONE"
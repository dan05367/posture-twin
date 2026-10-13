"""個人化校準與規則式姿勢判斷（第 1 週的 baseline，之後由 ML 模型取代）。"""
import json
import statistics
import time
from pathlib import Path

from features import FEATURE_NAMES

BASELINE_FILE = Path(__file__).with_name("calibration.json")

# 與「自己的標準坐姿」相比的偏離門檻，可依個人調整
THRESHOLDS = {
    "shoulder_tilt": 5.0,   # 度（絕對值）
    "head_roll": 8.0,       # 度（絕對值）
    "head_offset": 0.15,    # 肩寬倍數（絕對值）
    "head_drop": 0.12,      # 比基準「小」多少算低頭/駝背
    "shrug": 0.10,          # 比基準「小」多少算聳肩
    "lean": 1.15,           # 肩寬或臉寬放大倍率，超過算過度前傾
}


class Calibrator:
    """收集 duration 秒的特徵，取中位數當作基準。"""

    def __init__(self, duration: float = 5.0):
        self.duration = duration
        self.start = time.perf_counter()
        self.samples = {k: [] for k in FEATURE_NAMES}

    def add(self, feats: dict):
        for k in FEATURE_NAMES:
            self.samples[k].append(feats[k])

    @property
    def progress(self) -> float:
        return min((time.perf_counter() - self.start) / self.duration, 1.0)

    @property
    def done(self) -> bool:
        return self.progress >= 1.0

    def finish(self):
        n = len(next(iter(self.samples.values())))
        if n < 10:
            return None  # 樣本太少（人沒在畫面內）
        baseline = {k: statistics.median(v) for k, v in self.samples.items()}
        BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
        return baseline


def load_baseline():
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return None


def evaluate(feats: dict, base: dict):
    """回傳 (deviations, issues)。deviations 為各特徵相對基準的偏差。"""
    t = THRESHOLDS
    dev = {
        "shoulder_tilt": feats["shoulder_tilt"] - base["shoulder_tilt"],
        "head_roll": feats["head_roll"] - base["head_roll"],
        "head_offset": feats["head_offset"] - base["head_offset"],
        "head_drop": feats["head_drop"] - base["head_drop"],
        "shrug": feats["shrug"] - base["shrug"],
        "lean": max(feats["shoulder_w"] / base["shoulder_w"],
                    feats["face_w"] / base["face_w"]),
    }
    issues = []
    if abs(dev["shoulder_tilt"]) > t["shoulder_tilt"]:
        issues.append("uneven shoulders")
    if abs(dev["head_roll"]) > t["head_roll"]:
        issues.append("head tilted")
    if abs(dev["head_offset"]) > t["head_offset"]:
        issues.append("head off-center")
    if -dev["head_drop"] > t["head_drop"]:
        issues.append("slouching / head down")
    if -dev["shrug"] > t["shrug"]:
        issues.append("shoulders raised")
    if dev["lean"] > t["lean"]:
        issues.append("leaning toward screen")
    return dev, issues

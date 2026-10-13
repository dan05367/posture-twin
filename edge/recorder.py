"""錄製模式：把「特徵 + 標籤」寫成 CSV，只存數值，不存影像。

每次啟動程式 = 一個 session（一個 CSV）。訓練時會依 session 分組做交叉驗證，
所以請在不同時間、光線、坐姿位置各自重開程式錄製。
"""
import csv
import time
from datetime import datetime
from pathlib import Path

from features import FEATURE_NAMES

LABELS = ["good", "slouch", "lean_forward", "head_tilt", "shoulder_uneven"]
DEV_NAMES = ["shoulder_tilt", "head_roll", "head_offset", "head_drop", "shrug", "lean"]
COLUMNS = (["ts", "session", "label"]
           + FEATURE_NAMES
           + [f"dev_{k}" for k in DEV_NAMES])


class Recorder:
    def __init__(self, out_dir, hz: float = 10.0):
        self.out_dir = Path(out_dir)
        self.interval = 1.0 / hz
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.count = 0
        self._last = 0.0
        self._f = None
        self._w = None

    def _open(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"session_{self.session}.csv"
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)
        print(f"[recorder] writing to {path}")

    def write(self, label: str, feats: dict, dev: dict):
        now = time.perf_counter()
        if now - self._last < self.interval:
            return
        self._last = now
        if self._f is None:  # 第一次真的要寫才建檔，避免產生空檔案
            self._open()
        row = ([f"{time.time():.3f}", self.session, label]
               + [f"{feats[k]:.5f}" for k in FEATURE_NAMES]
               + [f"{dev[k]:.5f}" for k in DEV_NAMES])
        self._w.writerow(row)
        self._f.flush()
        self.count += 1

    def close(self):
        if self._f:
            self._f.close()

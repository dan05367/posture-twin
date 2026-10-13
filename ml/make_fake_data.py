"""產生假資料，用來在真正錄製前先驗證 MLflow + 訓練流程是否通。
輸出到 data/fake/（不要放進 data/raw，以免和真實資料混在一起）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "fake"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

COLS = ["dev_shoulder_tilt", "dev_head_roll", "dev_head_offset",
        "dev_head_drop", "dev_shrug", "dev_lean"]
NOISE = np.array([2.5, 3.0, 0.05, 0.05, 0.04, 0.05])
CENTER = {  # 各標籤的平均偏差（lean 是比例，good 時約為 1.0）
    "good":            [0, 0, 0, 0, 0, 1.0],
    "slouch":          [0, 0, 0, -0.2, -0.05, 1.05],
    "lean_forward":    [0, 0, 0, -0.03, 0, 1.25],
    "head_tilt":       [0, 12, 0.2, 0, 0, 1.0],
    "shoulder_uneven": [8, 0, 0, 0, 0, 1.0],
}

for s in range(6):
    shift = rng.normal(0, 1, 6) * NOISE * 0.3  # 模擬每個 session 的環境差異
    rows = []
    for label, c in CENTER.items():
        x = np.array(c) + shift + rng.normal(0, 1, (200, 6)) * NOISE
        d = pd.DataFrame(x, columns=COLS)
        d.insert(0, "label", label)
        rows.append(d)
    out = pd.concat(rows)
    out.insert(0, "session", f"fake_{s}")
    out.to_csv(OUT / f"session_fake_{s}.csv", index=False)
print(f"wrote 6 sessions to {OUT}")

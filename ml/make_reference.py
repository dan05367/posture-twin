"""從訓練資料產生 drift 監控用的 reference.csv（只含 label 與 dev_ 特徵，沒有影像）。

用法（從專案根目錄）：
    python ml/make_reference.py                      # 使用 data/raw
    python ml/make_reference.py --data data/fake     # 用假資料先試
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["dev_shoulder_tilt", "dev_head_roll", "dev_head_offset",
        "dev_head_drop", "dev_shrug", "dev_lean"]

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
ap.add_argument("--out", default=str(ROOT / "monitoring" / "reference" / "reference.csv"))
args = ap.parse_args()

files = sorted(Path(args.data).glob("*.csv"))
if not files:
    raise SystemExit(f"{args.data} 裡找不到 CSV")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[["label"] + COLS].dropna()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"wrote {len(df)} rows from {len(files)} files -> {out}")
print(df["label"].value_counts().to_string())

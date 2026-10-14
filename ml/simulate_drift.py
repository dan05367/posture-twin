"""模擬 drift：對 API 送出「姿勢良好、但整體偏移」的假資料，用來示範儀表板與警報。
偏移量刻意設在規則門檻以內（肩膀 < 5 度、頭部側傾 < 8 度），讓這些樣本仍被判為 good，
這樣 drift 服務（預設只比較 good 的樣本）才看得到它們。

用法：
    python ml/simulate_drift.py                    # 偏移的資料
    python ml/simulate_drift.py --shift-tilt 0 --shift-roll 0 --shift-offset 0   # 不偏移（對照組）
"""
import argparse
import json
import random
import time
import urllib.request
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--rate", type=float, default=20, help="每秒送幾筆")
ap.add_argument("--shift-tilt", type=float, default=3.5, help="肩膀傾斜角偏移（度）")
ap.add_argument("--shift-roll", type=float, default=4.0, help="頭部側傾偏移（度）")
ap.add_argument("--shift-offset", type=float, default=0.06, help="頭部水平偏移（肩寬倍數）")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = random.Random(args.seed)
seen = Counter()
for i in range(args.n):
    dev = {
        "shoulder_tilt": rng.gauss(args.shift_tilt, 1.5),
        "head_roll": rng.gauss(args.shift_roll, 2.0),
        "head_offset": rng.gauss(args.shift_offset, 0.03),
        "head_drop": rng.gauss(0, 0.03),
        "shrug": rng.gauss(0, 0.03),
        "lean": rng.gauss(1.0, 0.03),
    }
    req = urllib.request.Request(f"{args.api}/features", data=json.dumps({"deviations": dev}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        seen[json.load(r)["stable_label"]] += 1
    time.sleep(1 / args.rate)

print(f"sent {args.n} samples; stable labels returned by API: {dict(seen)}")

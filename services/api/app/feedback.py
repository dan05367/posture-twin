"""把使用者的更正存成 CSV（只有數值特徵與標籤）。格式與錄製的資料相容，訓練時會一起讀取。"""
import csv
import threading
from datetime import datetime
from pathlib import Path

from .schema import DEV_NAMES

COLUMNS = ["ts", "session", "label", "predicted"] + [f"dev_{k}" for k in DEV_NAMES]
_lock = threading.Lock()


def save_feedback(out_dir, label: str, rows: list, when: datetime | None = None):
    """把 rows 以 label 為標籤附加到當天的 feedback_YYYYMMDD.csv，回傳 (筆數, 路徑)。
    session 欄位用日期，訓練時同一天的更正會被視為同一組。"""
    when = when or datetime.now()
    session = "feedback_" + when.strftime("%Y%m%d")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session}.csv"
    with _lock:
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(COLUMNS)
            for r in rows:
                w.writerow([f"{r['ts']:.3f}", session, label, r["stable_label"]]
                           + [f"{r[k]:.5f}" for k in DEV_NAMES])
    return len(rows), path

"""何時該重新訓練（只用標準函式庫，方便測試）。

重新訓練的觸發條件是「累積了足夠的新標註資料」（錄製的新 session 或使用者按「其實是…」的更正）。
資料 drift 只會發出通知，不會直接觸發訓練：資料分佈變了但沒有新標註，重訓練也學不到新東西。
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TriggerConfig:
    min_new_rows: int = 100                       # 至少新增幾筆標註資料才重訓練
    min_interval_sec: float = 6 * 3600            # 兩次訓練之間至少間隔多久
    drift_notify_cooldown_sec: float = 6 * 3600   # drift 通知的冷卻時間


def count_rows(dirs) -> int:
    """計算資料夾裡所有 CSV 的資料筆數（不含標題列）。"""
    total = 0
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in p.glob("*.csv"):
            with open(f, "rb") as fh:
                total += max(sum(1 for _ in fh) - 1, 0)
    return total


def evaluate_triggers(total_rows: int, trained_rows: int, last_run_ts, now: float,
                      cfg: TriggerConfig, force: bool = False):
    """回傳 (要不要訓練, 原因)。"""
    if force:
        return True, "manual request"
    new_rows = total_rows - trained_rows
    if new_rows < cfg.min_new_rows:
        return False, f"only {new_rows}/{cfg.min_new_rows} new labeled rows"
    if last_run_ts is not None and now - last_run_ts < cfg.min_interval_sec:
        wait_min = (cfg.min_interval_sec - (now - last_run_ts)) / 60
        return False, f"{new_rows} new rows, cooling down for {wait_min:.0f} more minutes"
    return True, f"{new_rows} new labeled rows"


def should_notify_drift(drifting: bool, last_notify_ts, now: float, cooldown: float) -> bool:
    if not drifting:
        return False
    return last_notify_ts is None or now - last_notify_ts >= cooldown


class StateStore:
    """把狀態存成 JSON 檔（重啟後還記得上次訓練到哪）。"""

    def __init__(self, path):
        self.path = Path(path)

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def save(self, state: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.path)

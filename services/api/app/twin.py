"""Digital twin 的狀態：最新姿勢、投票後的穩定標籤、不良姿勢累積時間、最近 N 分鐘的分佈，
以及最近一批特徵（給 drift 監控服務取用）。
目前存在記憶體裡（單一程序），重啟會歸零；之後可換成 Redis 或資料庫。"""
import threading
import time
from collections import Counter, deque

from .schema import DEV_NAMES

GOOD = "good"


class TwinState:
    def __init__(self, window_sec: float = 600.0, vote: int = 5, max_gap: float = 5.0,
                 max_rows: int = 5000):
        self.window_sec = window_sec
        self.max_gap = max_gap          # 兩筆資料間隔超過這麼久就最多只算這麼多秒
        self._lock = threading.Lock()
        self._recent = deque(maxlen=vote)
        self._history = deque()         # (time, label)
        self._rows = deque(maxlen=max_rows)   # 最近的特徵列（只有數值）
        self._last = None
        self._prev_t = None
        self._bad_total = 0.0
        self._streak_start = None

    def update(self, label, probs, deviations, now=None) -> str:
        now = time.time() if now is None else now
        with self._lock:
            self._recent.append(label)
            counts = Counter(self._recent)
            top = max(counts.values())
            # 票數相同時，取最近出現的
            stable = next(l for l in reversed(self._recent) if counts[l] == top)

            if self._prev_t is not None and stable != GOOD:
                self._bad_total += min(now - self._prev_t, self.max_gap)
            if stable != GOOD:
                if self._streak_start is None:
                    self._streak_start = now
            else:
                self._streak_start = None
            self._prev_t = now

            self._history.append((now, label))
            cutoff = now - self.window_sec
            while self._history and self._history[0][0] < cutoff:
                self._history.popleft()

            self._rows.append({"ts": now, "label": label, "stable_label": stable,
                               **{k: float(deviations[k]) for k in DEV_NAMES if k in deviations}})

            self._last = {"label": label, "stable_label": stable,
                          "probabilities": probs, "deviations": deviations,
                          "updated_at": now}
            return stable

    def recent(self, limit: int = 1000, stable_label: str | None = None,
               since: float | None = None) -> list:
        """最近 limit 筆特徵（可只取某個穩定標籤、某個時間之後），由舊到新。"""
        with self._lock:
            rows = [r for r in self._rows
                    if (stable_label is None or r["stable_label"] == stable_label)
                    and (since is None or r["ts"] >= since)]
            return rows[-limit:]

    def snapshot(self, now=None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            if self._last is None:
                return {"has_data": False}
            counts = Counter(l for _, l in self._history)
            n = sum(counts.values())
            bad = n - counts.get(GOOD, 0)
            return {
                "has_data": True,
                **self._last,
                "age_sec": round(now - self._last["updated_at"], 2),
                "bad_seconds_total": round(self._bad_total, 1),
                "bad_streak_sec": round(now - self._streak_start, 1) if self._streak_start else 0.0,
                "window_sec": self.window_sec,
                "window_counts": dict(counts),
                "window_bad_ratio": round(bad / n, 3) if n else 0.0,
            }

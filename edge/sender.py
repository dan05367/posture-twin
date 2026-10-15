"""背景執行緒：每隔 interval 秒把最新的特徵 POST 到 API（只送數值，不送影像），
並記住 API 回傳的預測，讓畫面和通知可以使用模型的判斷。"""
import threading
import time

import requests


class Sender:
    def __init__(self, url: str, interval: float = 1.0):
        self.url = url
        self.interval = interval
        self._latest = None
        self._last_result = None          # (收到時間, API 回傳的 dict)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def update(self, payload: dict):
        with self._lock:
            self._latest = payload

    def latest(self, max_age: float = 3.0):
        """最近一次 API 的預測；超過 max_age 秒沒更新（API 沒開或斷線）就回傳 None。"""
        with self._lock:
            if self._last_result is None:
                return None
            t, result = self._last_result
        return result if time.monotonic() - t <= max_age else None

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                payload, self._latest = self._latest, None
            if payload is None:
                continue
            try:
                r = requests.post(self.url, json=payload, timeout=0.5)
                if r.ok:
                    with self._lock:
                        self._last_result = (time.monotonic(), r.json())
            except (requests.RequestException, ValueError):
                pass  # 服務沒開時不影響 Edge 運作

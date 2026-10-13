"""背景執行緒：每隔 interval 秒把最新的特徵 POST 到 API（只送數值，不送影像）。"""
import threading

import requests


class Sender:
    def __init__(self, url: str, interval: float = 1.0):
        self.url = url
        self.interval = interval
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def update(self, payload: dict):
        with self._lock:
            self._latest = payload

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                payload, self._latest = self._latest, None
            if payload is None:
                continue
            try:
                requests.post(self.url, json=payload, timeout=0.5)
            except requests.RequestException:
                pass  # 服務沒開時不影響 Edge 運作

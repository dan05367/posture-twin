"""通知：姿勢不良持續一段時間後提醒使用者。可同時使用多個管道：
  - console：印在終端機
  - desktop：Windows 桌面通知（需要 `pip install plyer`；沒裝就退回系統提示音）
  - ntfy：手機推播（手機裝 ntfy App，訂閱同一個主題名稱）
"""
import json
import time
import urllib.request


class Notifier:
    def __init__(self, backends, cooldown: float = 300.0, clock=time.monotonic):
        self.backends = list(backends)
        self.cooldown = cooldown        # 兩次通知之間至少間隔幾秒，避免一直吵
        self.clock = clock
        self._last = None

    def maybe_notify(self, title: str, message: str) -> bool:
        now = self.clock()
        if self._last is not None and now - self._last < self.cooldown:
            return False
        self._last = now
        for backend in self.backends:
            try:
                backend(title, message)
            except Exception as e:      # 通知失敗不能影響主程式
                print(f"[notify] {getattr(backend, '__name__', 'backend')} failed: {e}")
        return True


def console_backend(title: str, message: str):
    print(f"[notify] {title}: {message}")


def desktop_backend(title: str, message: str):
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=8)
        return
    except Exception:
        pass
    try:
        import winsound            # 只有 Windows 有
        winsound.MessageBeep()
    except Exception:
        pass


def make_ntfy_backend(topic: str, server: str = "https://ntfy.sh"):
    def ntfy_backend(title: str, message: str):
        req = urllib.request.Request(
            server.rstrip("/") + "/",
            data=json.dumps({"topic": topic, "title": title, "message": message}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "posture-twin"})
        urllib.request.urlopen(req, timeout=5).close()
    return ntfy_backend

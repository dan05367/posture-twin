"""通知：Discord / Slack webhook、ntfy 手機推播。用環境變數設定，沒設定的管道就跳過。

  DISCORD_WEBHOOK_URL   Discord 頻道 → 編輯頻道 → 整合 → Webhook
  SLACK_WEBHOOK_URL     Slack Incoming Webhook
  NTFY_TOPIC            ntfy 的主題名稱（手機裝 ntfy App 訂閱同一個主題；請取一個不易猜到的名稱）
  NTFY_SERVER           預設 https://ntfy.sh
（LINE Notify 已於 2025 年 3 月終止服務，所以不支援。）
"""
import json
import logging
import os
import urllib.request

log = logging.getLogger("notify")
HEADERS = {"Content-Type": "application/json", "User-Agent": "posture-twin/1.0"}  # Discord 會擋預設的 Python UA


def build_requests(title: str, message: str, env=None):
    env = os.environ if env is None else env
    out = []
    if env.get("DISCORD_WEBHOOK_URL"):
        out.append(("discord", env["DISCORD_WEBHOOK_URL"], {"content": f"**{title}**\n{message}"}))
    if env.get("SLACK_WEBHOOK_URL"):
        out.append(("slack", env["SLACK_WEBHOOK_URL"], {"text": f"*{title}*\n{message}"}))
    if env.get("NTFY_TOPIC"):
        server = env.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        out.append(("ntfy", server + "/", {"topic": env["NTFY_TOPIC"], "title": title, "message": message}))
    return out


def send(title: str, message: str, env=None, opener=urllib.request.urlopen) -> list:
    """送到所有已設定的管道，回傳成功的管道名稱。任何失敗只記 log，不會丟出例外。"""
    log.info("NOTIFY %s | %s", title, message)
    delivered = []
    for name, url, body in build_requests(title, message, env):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                         headers=HEADERS, method="POST")
            opener(req, timeout=5).close()
            delivered.append(name)
        except Exception as e:
            log.warning("notify via %s failed: %r", name, e)
    return delivered

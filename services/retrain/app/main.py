"""自動重新訓練服務（只用標準函式庫的 http.server）。

每隔 CHECK_INTERVAL_SEC 秒：
  1. 數一數 data/raw + data/feedback 有多少標註資料，比上次訓練時多了多少
  2. 夠多（且過了冷卻時間）→ 執行 ml/train.py --register-if-better
  3. 新模型通過把關並註冊 → 呼叫 API 的 /reload 上線；沒通過 → 保留舊模型
  4. 結果以 Discord / Slack / ntfy 通知；另外若 drift 服務回報資料 drift，也會通知（有冷卻時間）
端點：GET /health /metrics /status，POST /run（手動觸發）
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

from . import notify
from .triggers import StateStore, TriggerConfig, count_rows, evaluate_triggers, should_notify_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrain")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
DRIFT_URL = os.getenv("DRIFT_URL", "http://drift:8001").rstrip("/")
UI_URL = os.getenv("PUBLIC_UI_URL", "http://localhost:8000/ui")
DATA_DIRS = [d for d in os.getenv("DATA_DIRS", "/data/raw,/data/feedback").split(",") if d]
TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT", "/srv/ml/train.py")
STATE_PATH = os.getenv("STATE_PATH", "/state/retrain_state.json")
CHECK_INTERVAL = float(os.getenv("CHECK_INTERVAL_SEC", "600"))
TRAIN_TIMEOUT = float(os.getenv("TRAIN_TIMEOUT_SEC", "1800"))
MIN_F1 = os.getenv("MIN_F1", "0.7")
TOLERANCE = os.getenv("TOLERANCE", "0.02")
PORT = int(os.getenv("PORT", "8002"))
CFG = TriggerConfig(min_new_rows=int(os.getenv("MIN_NEW_ROWS", "100")),
                    min_interval_sec=float(os.getenv("MIN_INTERVAL_SEC", str(6 * 3600))),
                    drift_notify_cooldown_sec=float(os.getenv("DRIFT_NOTIFY_COOLDOWN_SEC", str(6 * 3600))))

RUNS = Counter("retrain_runs_total", "Retraining runs by outcome", ["outcome"])
PENDING = Gauge("retrain_pending_rows", "Labeled rows added since the last training")
RUNNING = Gauge("retrain_running", "1 while training is running")
LAST_TS = Gauge("retrain_last_run_timestamp_seconds", "Unix time of the last training run")
LAST_F1 = Gauge("retrain_last_macro_f1", "Macro F1 of the last training run")
for _o in ("registered", "rejected", "failed"):
    RUNS.labels(outcome=_o).inc(0)

STORE = StateStore(STATE_PATH)
STATE_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
MESSAGE = {"text": "starting", "ts": None}


def update_state(**kw) -> dict:
    with STATE_LOCK:
        s = STORE.load()
        s.update(kw)
        STORE.save(s)
        return s


def say(text: str):
    MESSAGE.update(text=text, ts=time.time())
    log.info(text)


def reload_api() -> bool:
    try:
        req = urllib.request.Request(f"{API_URL}/reload", method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status == 200
    except Exception as e:
        log.warning("API reload failed: %r", e)
        return False


def run_training(reason: str) -> dict:
    out = Path(tempfile.gettempdir()) / f"retrain_summary_{int(time.time())}.json"
    cmd = [sys.executable, TRAIN_SCRIPT, "--data", *DATA_DIRS, "--tracking-uri", MLFLOW_URI,
           "--register-if-better", "--min-f1", MIN_F1, "--tolerance", TOLERANCE,
           "--reason", reason, "--summary-json", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TRAIN_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-600:])
    return json.loads(out.read_text(encoding="utf-8"))


def attempt(force: bool = False) -> dict:
    if not RUN_LOCK.acquire(blocking=False):
        return {"status": "busy"}
    try:
        total = count_rows(DATA_DIRS)
        state = STORE.load()
        if "trained_rows" not in state:
            # 第一次啟動：假設現有資料已經用來訓練過了，從現在開始只計算「新增」的資料
            state = update_state(trained_rows=total)
            say(f"baseline set: {total} existing rows")
        PENDING.set(max(total - state["trained_rows"], 0))

        go, why = evaluate_triggers(total, state["trained_rows"], state.get("last_run_ts"),
                                    time.time(), CFG, force)
        if not go:
            say(f"no retraining: {why}")
            return {"status": "skipped", "reason": why}

        say(f"retraining: {why}")
        RUNNING.set(1)
        now = time.time()
        try:
            summary = run_training(why)
        except Exception as e:
            RUNS.labels(outcome="failed").inc()
            LAST_TS.set(now)
            update_state(last_run_ts=now, last_result={"status": "failed", "error": repr(e), "ts": now})
            notify.send("重新訓練失敗", f"原因：{why}\n錯誤：{str(e)[-300:]}")
            return {"status": "failed", "error": repr(e)}

        registered = bool(summary["registered"])
        RUNS.labels(outcome="registered" if registered else "rejected").inc()
        LAST_TS.set(now)
        LAST_F1.set(summary["best_macro_f1"])
        reloaded = reload_api() if registered else False
        update_state(trained_rows=total, last_run_ts=now,
                     last_result={"status": "registered" if registered else "rejected", "ts": now,
                                  "reloaded": reloaded, **summary})
        if registered:
            notify.send("新模型已上線",
                        f"版本 {summary['model_version']}，macro F1 {summary['best_macro_f1']:.3f}"
                        f"（規則式 {summary['rule_macro_f1']:.3f}）\n觸發原因：{why}"
                        + ("" if reloaded else "\n⚠ API 重新載入失敗，請手動呼叫 POST /reload"))
        else:
            notify.send("新模型未通過把關，保留原模型",
                        f"macro F1 {summary['best_macro_f1']:.3f}\n原因：{summary['reason']}")
        return {"status": "registered" if registered else "rejected", **summary}
    finally:
        RUNNING.set(0)
        RUN_LOCK.release()


def check_drift():
    try:
        with urllib.request.urlopen(f"{DRIFT_URL}/status", timeout=5) as r:
            st = json.load(r)
    except Exception:
        return
    drifting = st.get("status") == "ok" and bool((st.get("result") or {}).get("dataset_drift"))
    now = time.time()
    if should_notify_drift(drifting, STORE.load().get("last_drift_notify_ts"), now, CFG.drift_notify_cooldown_sec):
        update_state(last_drift_notify_ts=now)
        notify.send("偵測到資料 drift",
                    "輸入資料的分佈和訓練資料明顯不同（鏡頭角度、光線或校準可能改變）。\n"
                    f"建議：重新校準，並在 {UI_URL} 對判斷錯誤的姿勢按「其實是…」補充標註；"
                    "累積足夠的新標註後會自動重新訓練。")


def loop():
    while True:
        try:
            attempt()
            check_drift()
        except Exception:
            log.exception("retrain loop error")
        time.sleep(CHECK_INTERVAL)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/health":
            self._send(200, json.dumps({"status": "ok"}))
        elif path == "/metrics":
            self._send(200, generate_latest(), CONTENT_TYPE_LATEST)
        elif path == "/status":
            body = {"running": RUN_LOCK.locked(), "message": MESSAGE, "state": STORE.load(),
                    "config": {"min_new_rows": CFG.min_new_rows, "min_interval_sec": CFG.min_interval_sec,
                               "data_dirs": DATA_DIRS}}
            self._send(200, json.dumps(body, indent=2, default=str))
        else:
            self._send(404, json.dumps({"detail": "not found"}))

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/run":
            return self._send(404, json.dumps({"detail": "not found"}))
        if RUN_LOCK.locked():
            return self._send(409, json.dumps({"detail": "training already running"}))
        threading.Thread(target=attempt, kwargs={"force": True}, daemon=True).start()
        self._send(202, json.dumps({"detail": "retraining started; see /status"}))

    def log_message(self, *args):
        pass


def main():
    threading.Thread(target=loop, daemon=True).start()
    log.info("retrain service on :%d (mlflow=%s, data=%s)", PORT, MLFLOW_URI, DATA_DIRS)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

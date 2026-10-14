"""Drift 監控服務（只用標準函式庫的 http.server，依賴越少越穩）。

每隔 CHECK_INTERVAL_SEC 秒：向 API 取最近的特徵 → 與訓練時的 reference 比較 → 更新 Prometheus 指標。
端點：/health  /metrics  /status  /report（Evidently 的 HTML 報告）
"""
import json
import logging
import math
import os
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

from .core import DEV_COLS, compute_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drift")

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "/reference/reference.csv")
CHECK_INTERVAL = float(os.getenv("CHECK_INTERVAL_SEC", "30"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "1000"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "100"))
MIN_REFERENCE = int(os.getenv("MIN_REFERENCE", "50"))
FILTER_LABEL = os.getenv("DRIFT_FILTER_LABEL", "good")   # 空字串 = 不過濾
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.5"))
PORT = int(os.getenv("PORT", "8001"))

FEATURES = [c.removeprefix("dev_") for c in DEV_COLS]
NAN = float("nan")

PSI = Gauge("posture_drift_psi", "PSI of each feature vs reference", ["feature"])
KS = Gauge("posture_drift_ks_statistic", "KS statistic of each feature vs reference", ["feature"])
DETECTED = Gauge("posture_drift_detected", "1 if the feature drifted", ["feature"])
SHARE = Gauge("posture_drift_share", "Share of drifted features")
DATASET = Gauge("posture_dataset_drift", "1 if dataset-level drift detected")
N_CUR = Gauge("posture_drift_current_samples", "Samples in the current window")
N_REF = Gauge("posture_drift_reference_samples", "Samples in the reference set")
LAST = Gauge("posture_drift_last_check_timestamp_seconds", "Unix time of the last successful check")
ERRORS = Counter("posture_drift_check_errors_total", "Failed drift checks")

LOCK = threading.Lock()
STATE = {"status": "starting", "detail": "", "last_check": None, "result": None}
CACHE = {"ref": None, "mtime": None, "cur": None}


def clear_gauges():
    """沒有有效結果時把 drift 指標設成 NaN，避免舊的 drift=1 一直掛著觸發警報。"""
    SHARE.set(NAN)
    DATASET.set(NAN)
    for f in FEATURES:
        PSI.labels(feature=f).set(NAN)
        KS.labels(feature=f).set(NAN)
        DETECTED.labels(feature=f).set(NAN)


def set_state(status: str, detail: str = "", result=None):
    with LOCK:
        STATE.update(status=status, detail=detail, last_check=time.time(), result=result)
    log.info("status=%s %s", status, detail)


def load_reference():
    p = Path(REFERENCE_PATH)
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    if CACHE["mtime"] != mtime:
        df = pd.read_csv(p)
        if FILTER_LABEL and "label" in df.columns:
            df = df[df["label"] == FILTER_LABEL]
        CACHE.update(ref=df.reset_index(drop=True), mtime=mtime)
    return CACHE["ref"]


def fetch_current() -> pd.DataFrame:
    q = {"limit": WINDOW_SIZE}
    if FILTER_LABEL:
        q["stable_label"] = FILTER_LABEL
    with urllib.request.urlopen(f"{API_URL}/recent?{urllib.parse.urlencode(q)}", timeout=5) as r:
        rows = json.load(r)["rows"]
    df = pd.DataFrame(rows)
    return df.rename(columns={f: f"dev_{f}" for f in FEATURES})


def check_once():
    try:
        ref = load_reference()
        if ref is None:
            clear_gauges()
            return set_state("no_reference", f"{REFERENCE_PATH} not found; run ml/make_reference.py")
        N_REF.set(len(ref))
        if len(ref) < MIN_REFERENCE:
            clear_gauges()
            return set_state("reference_too_small", f"{len(ref)}/{MIN_REFERENCE} rows after filtering")

        try:
            cur = fetch_current()
        except Exception as e:
            ERRORS.inc()
            clear_gauges()
            return set_state("api_unreachable", str(e))

        N_CUR.set(len(cur))
        if len(cur) < MIN_SAMPLES:
            clear_gauges()
            return set_state("waiting_for_data", f"{len(cur)}/{MIN_SAMPLES} samples")

        res = compute_drift(ref, cur, DEV_COLS, PSI_THRESHOLD, SHARE_THRESHOLD)
        for col, f in res["features"].items():
            name = col.removeprefix("dev_")
            PSI.labels(feature=name).set(f["psi"])
            KS.labels(feature=name).set(f["ks_statistic"])
            DETECTED.labels(feature=name).set(1.0 if f["drifted"] else 0.0)
        SHARE.set(res["share"])
        DATASET.set(1.0 if res["dataset_drift"] else 0.0)
        LAST.set(time.time())
        CACHE["cur"] = cur
        set_state("ok", f"{res['n_drifted']}/{len(DEV_COLS)} features drifted", res)
    except Exception as e:
        ERRORS.inc()
        clear_gauges()
        log.exception("drift check failed")
        set_state("error", repr(e))


def loop():
    while True:
        check_once()
        time.sleep(CHECK_INTERVAL)


def _clean(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def build_report_html() -> str:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    ref, cur = CACHE["ref"], CACHE["cur"]
    if ref is None or cur is None:
        raise RuntimeError("no data yet: wait until /status shows status=ok")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref[DEV_COLS], current_data=cur[DEV_COLS])
    return report.get_html()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str = "application/json"):
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
            with LOCK:
                self._send(200, json.dumps(_clean(STATE), default=str, indent=2))
        elif path == "/report":
            try:
                self._send(200, build_report_html(), "text/html; charset=utf-8")
            except Exception as e:
                self._send(503, f"report unavailable: {e!r}", "text/plain; charset=utf-8")
        else:
            self._send(404, json.dumps({"detail": "not found"}))

    def log_message(self, *args):  # 不要每個請求都印 log
        pass


def main():
    clear_gauges()
    threading.Thread(target=loop, daemon=True).start()
    log.info("drift service listening on :%d (api=%s, reference=%s)", PORT, API_URL, REFERENCE_PATH)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

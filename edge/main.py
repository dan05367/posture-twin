"""Edge 端主程式（Windows 原生執行）：鏡頭 -> MediaPipe -> 特徵 -> 平滑 -> 判斷 -> 顯示。

按鍵：  c = 開始校準（請坐好、看正前方 5 秒）   r = 清除校準   q = 離開
        1~5 = 選擇標籤（good/slouch/lean_forward/head_tilt/shoulder_uneven）
        空白鍵 = 開始/暫停錄製（需先校準；轉換姿勢時請先暫停）
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

from features import FEATURE_NAMES, extract
from one_euro_filter import OneEuroFilter
from posture import BASELINE_FILE, Calibrator, evaluate, load_baseline
from recorder import LABELS, Recorder
from sender import Sender

GREEN, YELLOW, RED, WHITE = (0, 200, 0), (0, 220, 255), (0, 0, 255), (255, 255, 255)
WARN_AFTER_SEC = 3.0  # 姿勢不良持續幾秒才警告


def put(img, text, y, color=WHITE):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--api-url", default=None, help="例如 http://localhost:8000/features")
    ap.add_argument("--record-dir", default= "data")
    ap.add_argument("--record-hz", type=float, default=10.0, help="錄製頻率（每秒幾筆）")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # Windows 上 DSHOW 開啟較快
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("打不開鏡頭，請試試 --camera 1")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawer = mp.solutions.drawing_utils
    filters = {k: OneEuroFilter(min_cutoff=1.0, beta=0.05) for k in FEATURE_NAMES}
    sender = Sender(args.api_url) if args.api_url else None

    baseline = load_baseline()
    calib = None
    bad_since = None
    fps_t, fps = time.perf_counter(), 0.0
    recorder = Recorder(args.record_dir, args.record_hz)
    label = None
    rec_on = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # 鏡像顯示，比較直覺
        h, w = frame.shape[:2]
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
        fps_t = now

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feats = None
        if res.pose_landmarks:
            drawer.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            raw = extract(res.pose_landmarks.landmark, w, h)
            if raw:
                feats = {k: filters[k](v, now) for k, v in raw.items()}

        y = 22
        put(frame, f"FPS {fps:4.1f}   [c] calibrate  [r] reset  [q] quit", y)
        put(frame, "1 good 2 slouch 3 lean 4 head_tilt 5 sh_uneven  [space] rec", h - 36)
        put(frame, f"label: {label or '-'}   " + (f"REC {recorder.count}" if rec_on else "rec off"),
            h - 12, RED if rec_on else WHITE)

        if feats is None:
            bad_since = None
            put(frame, "No person detected (need face + both shoulders)", y + 24, YELLOW)
        elif calib is not None:
            calib.add(feats)
            put(frame, f"Calibrating... sit straight  {calib.progress * 100:3.0f}%", y + 24, YELLOW)
            if calib.done:
                baseline = calib.finish()
                calib = None
        elif baseline is None:
            put(frame, "Not calibrated. Press [c] while sitting in good posture.", y + 24, YELLOW)
        else:
            dev, issues = evaluate(feats, baseline)
            if issues:
                bad_since = bad_since or now
            else:
                bad_since = None
            bad_for = (now - bad_since) if bad_since else 0.0

            if not issues:
                put(frame, "Posture: GOOD", y + 24, GREEN)
            elif bad_for < WARN_AFTER_SEC:
                put(frame, f"Posture: checking... ({', '.join(issues)})", y + 24, YELLOW)
            else:
                put(frame, f"WARNING ({bad_for:.0f}s): {', '.join(issues)}", y + 24, RED)

            for i, (k, v) in enumerate(dev.items()):
                put(frame, f"{k:14s} {v:+7.2f}", y + 54 + i * 20)

            if sender:
                sender.update({"ts": time.time(), "features": feats,
                               "deviations": dev, "issues": issues})

            if rec_on and label:
                recorder.write(label, feats, dev)

        cv2.imshow("Posture Twin - Edge", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("1") <= key < ord("1") + len(LABELS):
            label = LABELS[key - ord("1")]
        if key == ord(" "):
            rec_on = not rec_on
        if key == ord("c"):
            calib = Calibrator(duration=5.0)
        if key == ord("r"):
            baseline = None
            BASELINE_FILE.unlink(missing_ok=True)

    recorder.close()
    if sender:
        sender.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

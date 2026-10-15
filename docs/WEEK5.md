# Week 5+：回饋迴路、自動重新訓練、Digital Twin 畫面、通知

## 整體流程
```
Edge 送特徵 → API 預測 → 網頁 /ui 顯示虛擬人偶
                 │              │
                 │              └─ 判斷錯了？按「其實是…」→ data/feedback/*.csv（新標註）
                 ▼
          drift 服務 ── 資料分佈變了 ──→ 通知你（建議補標註）
                                              │
retrain 服務：新標註 ≥ 100 筆且過了冷卻時間 → ml/train.py --register-if-better
                 ├─ 通過把關 → 註冊新版本 → 呼叫 API /reload 上線 → 通知
                 └─ 沒通過   → 保留舊模型 → 通知原因
```

## 檔案位置（對照你的 MLProject/ 結構）
```
MLProject/
├── .dockerignore                         ← 新增（retrain 以根目錄為 build context）
├── docker-compose.yml                    ← 覆蓋（新增 retrain；api 多掛載 data/feedback）
├── .github/workflows/ci.yml              ← 覆蓋
├── .gitignore                            ← 手動加一行 data/feedback/
├── edge/
│   ├── main.py, sender.py, requirements.txt   ← 覆蓋
│   ├── notifier.py                       ← 新增
│   └── tests/test_notifier.py            ← 新增
├── ml/
│   ├── train.py                          ← 覆蓋（v2）
│   ├── gating.py, pytest.ini             ← 新增
│   └── tests/test_gating.py              ← 新增
├── monitoring/prometheus/{prometheus,alerts}.yml   ← 覆蓋
└── services/
    ├── api/app/{main,schema,twin}.py     ← 覆蓋
    ├── api/app/feedback.py               ← 新增
    ├── api/app/static/twin.html          ← 新增（虛擬人偶網頁）
    ├── api/tests/{test_feedback,test_feedback_store}.py  ← 新增
    └── retrain/                          ← 整個新增
```

## 1. 啟動
```bash
cd ~/MLProject
mkdir -p data/feedback                    # 一定要先建，否則 Docker 會建成 root 擁有、API 寫不進去
echo "data/feedback/" >> .gitignore
docker compose up -d --build
```
- 虛擬人偶畫面：**http://localhost:8000/ui**
- 重新訓練服務狀態：http://localhost:8002/status

## 2. Digital twin 畫面
- 左邊的人偶會跟著你的偏差即時動：肩膀傾斜、頭部側傾、頭部水平偏移、低頭、聳肩、前傾（整個人偶放大）。顏色代表目前的穩定標籤。
- 因為是正面鏡頭、沒有可靠的深度，這是**正面 2D 人偶**，不是 3D 骨架。要升級成 Three.js 的話，資料來源（`GET /twin`）不用改。

## 3. 回饋迴路
模型判斷錯的時候，在網頁下方按「其實是：…」，會把**最近 10 秒**的資料以你指定的標籤存進 `data/feedback/feedback_日期.csv`（同時記錄模型當時預測了什麼，方便之後分析哪兩類最容易混淆）。連按兩次不會重複存。

## 4. 自動重新訓練
- 啟動時，現有的資料視為「已訓練過」，之後只計算**新增**的標註（新錄的 session、使用者更正）。
- 新增 ≥ `MIN_NEW_ROWS`（預設 100 筆）且距離上次訓練超過 `MIN_INTERVAL_SEC`（預設 6 小時）才會訓練。
- 手動觸發（忽略筆數與冷卻）：
  ```bash
  curl -X POST localhost:8002/run
  curl localhost:8002/status
  ```
- **把關規則**（`ml/gating.py`）：macro F1 ≥ 0.7、必須勝過規則式 baseline、不能比目前模型差超過 0.02，才會註冊新版本並讓 API 重新載入；否則保留舊模型，並通知你原因。
- **drift 只通知，不直接觸發訓練**：分佈變了但沒有新標註，重訓練也學不到新東西。

### 把關的限制
新模型和舊模型用的資料不同（新模型多了新標註），所以這只是「不退步」的把關，不是嚴格的 A/B 比較。想更嚴謹，要另外留一份固定的 hold-out 測試集，兩個模型都在上面評估。

## 5. 通知

**伺服器端**（重新訓練完成／失敗／被拒絕、偵測到 drift）：在 `MLProject/.env` 設定，沒設的管道就不會發送。
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
NTFY_TOPIC=一個不易猜到的名字，例如 posture-dan-8f3k2
# SLACK_WEBHOOK_URL=...
```
- Discord：頻道設定 → 整合 → Webhook → 複製網址
- ntfy：手機裝 ntfy App，訂閱同一個主題名稱（任何知道名稱的人都能收發，所以名稱要取得難猜）
- LINE Notify 已在 2025 年 3 月終止服務，所以沒有支援
- 改完 `.env` 後：`docker compose up -d retrain`

**Edge 端**（姿勢不良持續超過 N 秒就提醒你）：
```cmd
pip install plyer
python main.py --api-url http://localhost:8000/features --notify-after 0 --ntfy-topic posture-dan-8f3k2
```
- `--notify-after 0` 關閉通知；`--notify-cooldown 300` 兩次通知至少間隔 5 分鐘；`--no-desktop-notify` 不跳桌面通知
- Edge 現在會優先用 API 的模型預測（畫面右側標示 `[model 0.83]`），API 沒開時退回規則式（標示 `[rules]`）

## 6. 測試
```bash
cd services/api     && pytest -q    # 24 passed
cd ../retrain       && pytest -q    # 10 passed
cd ../../ml         && pytest -q    # 5 passed
cd ../edge          && pytest -q    # 9 passed
```

## 之後可以再做的
- 用固定 hold-out 測試集做嚴謹的新舊模型比較
- 把 `/ui` 換成 Three.js 3D 骨架（需要側面鏡頭或第二個視角才有可靠深度）
- 接 Alertmanager，把 Prometheus 的警報也送到 Discord / ntfy
- 時間窗口特徵或 LSTM（連續 N 秒的姿勢趨勢）
- 想用 DVC 管 `data/feedback`：先從 `.gitignore` 移除該行，再 `dvc add data/feedback`

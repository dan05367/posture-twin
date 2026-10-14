# Week 4：Prometheus + Grafana 監控與 drift 偵測

## 新增 / 修改的檔案（對應你的專案結構）
```
MLProject/
├── docker-compose.yml                      ← 覆蓋（新增 drift、prometheus、grafana）
├── .github/workflows/ci.yml                ← 覆蓋
├── .gitignore                              ← 加一行 monitoring/reference/*.csv
├── ml/
│   ├── make_reference.py                   ← 新增
│   └── simulate_drift.py                   ← 新增
├── monitoring/
│   ├── prometheus/{prometheus.yml, alerts.yml}
│   ├── grafana/provisioning/datasources/prometheus.yml
│   ├── grafana/provisioning/dashboards/dashboards.yml
│   ├── grafana/dashboards/posture.json
│   └── reference/.gitkeep
└── services/
    ├── api/app/{main.py, twin.py}          ← 覆蓋（新增指標與 /recent）
    ├── api/tests/{test_api.py, test_twin.py} ← 覆蓋（多了新測試）
    └── drift/                              ← 整個新增
        ├── app/{__init__,core,main}.py
        ├── tests/test_core.py
        └── Dockerfile, requirements*.txt, pytest.ini, .dockerignore
```

## 1. 建立 reference（drift 比較的基準）
```bash
cd ~/MLProject
python ml/make_reference.py              # 用 data/raw（真實資料）
# 或先用假資料： python ml/make_reference.py --data data/fake
```
reference 預設只取 `label == good` 的樣本，因為 drift 服務拿來比較的也是「API 判為 good 的即時樣本」。

## 2. 啟動
```bash
docker compose up -d --build
docker compose ps
```

| 服務 | 網址 |
|---|---|
| Grafana（帳密 admin / admin，第一次登入會要求改密碼） | http://localhost:3000 → Dashboards → Posture Twin |
| Prometheus（Status → Targets 看 api、drift 是否 UP；Alerts 看警報） | http://localhost:9090 |
| Drift 狀態 | http://localhost:8001/status |
| Drift 報告（Evidently HTML） | http://localhost:8001/report |

## 3. 示範 drift
先讓 Edge 或腳本送一批「正常」資料，再送「偏移」資料：
```bash
python ml/simulate_drift.py --shift-tilt 0 --shift-roll 0 --shift-offset 0   # 對照組
python ml/simulate_drift.py                                                  # 偏移的資料
```
等 30 秒內 drift 服務的下一次檢查，Grafana 的 **Dataset drift** 會變成 DRIFT，PSI 面板的曲線會越過紅線。
想清掉模擬資料：`docker compose restart api`。

## 4. 看得到什麼

| 指標 | 意思 |
|---|---|
| `posture_predictions_total{label}` | 各姿勢預測次數 |
| `posture_inference_seconds` | 推論延遲 |
| `posture_confidence` | 模型信心值分佈（持續偏低代表模型沒把握） |
| `posture_deviation{feature}` | 即時偏差值 |
| `posture_window_bad_ratio` | 最近 10 分鐘不良姿勢比例 |
| `posture_seconds_since_last_sample` | 距離上次收到 Edge 資料幾秒 |
| `posture_drift_psi{feature}` | 各特徵與 reference 的 PSI |
| `posture_dataset_drift` | 整體是否 drift（1 = 是） |

警報（`alerts.yml`）：ApiDown、NoSensorData、ModelFallback、DataDrift、PoorPostureSustained。目前沒有接 Alertmanager，只會顯示在 Prometheus 的 Alerts 頁面。

## 5. Drift 判定方式與限制
- 每個特徵用 **PSI** 判斷：>= 0.2 算該特徵 drift（< 0.1 穩定，0.1～0.25 輕微，> 0.25 明顯）。KS 統計量與 p 值一併輸出當參考。
- drift 的特徵 >= 50% 才算整體 drift。
- **只比較 good 的樣本**：訓練資料是刻意擺出各種姿勢，即時資料大多是 good，直接比整個資料集會永遠報 drift。比較 good 對 good，才能抓到鏡頭角度、光線、校準過期這類環境變化。
- 代價：偏移大到讓預測變成不良姿勢的樣本會被濾掉，這種情況要看 `posture_window_bad_ratio` 和各類別預測次數的變化。
- 想調整：改 `docker-compose.yml` 裡 drift 的環境變數（`PSI_THRESHOLD`、`DRIFT_FILTER_LABEL`，設成空字串就不過濾）。
- 資料少於 `MIN_SAMPLES`（預設 100 筆）不判定，指標會顯示為空值，避免舊結果一直觸發警報。

## 6. 本機測試
```bash
cd services/drift && pip install -r requirements-dev.txt && pytest -q     # 6 passed
cd ../api && pip install -r requirements.txt -r requirements-dev.txt && pytest -q   # 17 passed
```

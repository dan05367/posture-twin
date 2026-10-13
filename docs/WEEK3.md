# Week 3：FastAPI + Docker + GitHub Actions

## 專案結構（repo 根目錄 = posture-twin/）
```
posture-twin/
├── .github/workflows/ci.yml     ← CI 必須在 repo 根目錄
├── docker-compose.yml           ← mlflow + api
├── edge/                        ← 新增 tests/、pytest.ini
└── services/
    ├── api/                     ← 新增：FastAPI 服務
    │   ├── app/{main,schema,predictor,twin}.py
    │   ├── tests/
    │   ├── Dockerfile
    │   └── requirements.txt
    └── training/                ← requirements 把 scikit-learn 固定為 1.5.2
```

## 1. 讓訓練環境與 API 的版本一致（重要）
模型是 pickle 存的，scikit-learn 版本不同可能載入失敗或結果不同。
```bash
cd services/training
pip install scikit-learn==1.5.2
python train.py --data <你的資料夾> --register   # 重新訓練並註冊
```

## 2. 啟動 MLflow + API
```bash
cd ~/posture-twin
docker compose up -d --build
docker compose logs -f api          # 看有沒有 "loaded model from models:/..."
```
- API 文件（可直接在網頁試打）：http://localhost:8000/docs
- 沒註冊過模型也能啟動，會自動退回規則式；`/health` 的 `model_source` 會顯示目前用哪個
- 訓練並註冊新版本之後：`curl -X POST localhost:8000/reload`

## 3. 手動測試
```bash
curl localhost:8000/health
curl -X POST localhost:8000/features -H 'content-type: application/json' \
  -d '{"deviations":{"shoulder_tilt":0,"head_roll":0,"head_offset":0,"head_drop":-0.3,"shrug":0,"lean":1.0}}'
curl localhost:8000/twin
curl localhost:8000/metrics
```

## 4. 接上 Edge（Windows）
```cmd
python main.py --api-url http://localhost:8000/features
```
Edge 每秒送一次偏差值，然後瀏覽器開 http://localhost:8000/twin 就能看到 twin 的即時狀態。

## 5. 本機跑測試
```bash
cd services/api && pip install -r requirements.txt -r requirements-dev.txt && pytest -q
cd ../../edge && pip install pytest && pytest -q
```

## 6. GitHub Actions
```bash
git add . && git commit -m "week 3: api, docker, ci"
git remote add origin git@github.com:<你>/posture-twin.git
git push -u origin main
```
到 GitHub 的 **Actions** 分頁看結果。流程：edge 測試 + api 測試 → 建置 Docker image → 啟動容器做 smoke test → （只有 push 到 main 時）推到 GHCR（ghcr.io/<你>/posture-twin-api）。

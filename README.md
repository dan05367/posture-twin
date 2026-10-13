# Edge（Windows 原生執行）

```powershell
cd posture-twin\edge
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

1. 坐好、看正前方，按 `c`，保持 5 秒完成校準（存成 calibration.json）
2. 之後畫面會顯示各特徵相對你標準坐姿的偏差，姿勢不良超過 3 秒會警告
3. 之後接上 API：`python main.py --api-url http://localhost:8000/features`

校準檔 calibration.json 屬於個人資料，記得加進 .gitignore。

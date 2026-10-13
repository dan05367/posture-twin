# Week 2：錄製資料 → 訓練 → MLflow 追蹤

## 0. 專案放哪裡
建議把整個 `posture-twin/` 放在 **WSL 的檔案系統**（例如 `~/posture-twin`），
Windows 端用 `\\wsl$\Ubuntu\home\<你>\posture-twin\edge` 開啟並執行 Edge。
這樣 Edge 錄的 CSV 直接出現在 WSL 裡，訓練和 DVC 都不用搬檔案。

## 1. 啟動 MLflow（WSL）
```bash
cd ~/posture-twin
docker compose up -d
# 瀏覽器開 http://localhost:5000
```

## 2. 先用假資料驗證流程（WSL）
```bash
cd services/training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python make_fake_data.py
python train.py --data ../../data/fake
```
到 http://localhost:5000 應該看到一個 `compare` run，底下有 rule_based、logreg、random_forest、final_* 四個子 run。

## 3. 錄製真實資料（Windows）
```powershell
cd edge
.venv\Scripts\activate
python main.py
```
操作：`c` 校準 → 按 `1~5` 選標籤 → 空白鍵開始錄 → 做出該姿勢 → 空白鍵暫停 → 換下一個標籤。

| 鍵 | 標籤 | 怎麼做 |
|---|---|---|
| 1 | good | 你的標準坐姿，並加入自然小動作（喝水、轉頭看一下） |
| 2 | slouch | 駝背、頭往下沉，輕度到嚴重都要有 |
| 3 | lean_forward | 身體或頭靠近螢幕 |
| 4 | head_tilt | 歪頭，左右兩邊都錄 |
| 5 | shoulder_uneven | 一肩高一肩低，左右兩邊都錄 |

**資料量與多樣性（比數量更重要）**
- 每個標籤每個 session 約 1～2 分鐘（10 Hz ≈ 600～1200 筆）
- 至少 **5 個 session**：每次**重開程式**、重新校準，最好分不同時間、光線、椅子高度、鏡頭角度
- 每個姿勢要有「輕微 / 中等 / 明顯」不同程度
- 一個 session = 一個 CSV，訓練時整個 session 會一起被留作驗證，才能測出對新環境的泛化能力

## 4. 用真實資料訓練
```bash
python train.py --data ../../data/raw --register
```

## 5. 資料版本控管（DVC）
```bash
pip install dvc
cd ~/posture-twin
git init && dvc init
dvc add data/raw
git add data/raw.dvc data/.gitignore .gitignore && git commit -m "add raw posture data v1"
```
之後每次新增資料就 `dvc add data/raw` 再 commit。MLflow 裡的 `data_hash` 參數可以對照是哪一版資料訓練出來的。

## 結果怎麼看
- 先看 `rule_based` 與其他模型的 `macro_f1`：ML 模型沒有明顯勝過規則式，代表資料或特徵還不夠好
- 看 `confusion_matrix.png`，找出容易混淆的類別（常見：slouch 和 lean_forward）
- 看 `class_counts.json` 確認各類別樣本數是否均衡

"""Drift 計算（只依賴 numpy / pandas / scipy，方便單元測試）。

- PSI（Population Stability Index）：< 0.1 穩定，0.1~0.25 輕微變化，> 0.25 明顯變化
- KS 檢定：同時提供統計量與 p 值，當作輔助參考
判定「某個特徵 drift」以 PSI >= psi_threshold 為準；
當 drift 的特徵比例 >= share_threshold，視為整體資料 drift。
"""
import numpy as np
import pandas as pd
from scipy import stats

DEV_COLS = ["dev_shoulder_tilt", "dev_head_roll", "dev_head_offset",
            "dev_head_drop", "dev_shrug", "dev_lean"]


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    # 以 reference 的分位數切箱，兩端開放（超出範圍的值歸入頭尾箱）
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1]
    n_bins = len(inner) + 1
    ref_pct = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=n_bins) / len(ref)
    cur_pct = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=n_bins) / len(cur)
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, cols=DEV_COLS,
                  psi_threshold: float = 0.2, share_threshold: float = 0.5) -> dict:
    features = {}
    for col in cols:
        r, c = ref_df[col].dropna(), cur_df[col].dropna()
        p = psi(r, c)
        ks = stats.ks_2samp(r, c) if len(r) and len(c) else None
        features[col] = {
            "psi": p,
            "ks_statistic": float(ks.statistic) if ks else float("nan"),
            "ks_pvalue": float(ks.pvalue) if ks else float("nan"),
            "drifted": bool(p == p and p >= psi_threshold),   # p == p 排除 NaN
        }
    n_drifted = sum(f["drifted"] for f in features.values())
    share = n_drifted / len(cols) if cols else 0.0
    return {
        "features": features,
        "n_drifted": int(n_drifted),
        "share": float(share),
        "dataset_drift": bool(share >= share_threshold),
        "n_reference": int(len(ref_df)),
        "n_current": int(len(cur_df)),
        "psi_threshold": psi_threshold,
        "share_threshold": share_threshold,
    }

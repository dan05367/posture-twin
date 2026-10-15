"""新模型要不要註冊（上線）的判斷規則。只用標準函式庫，方便測試。"""


def decide_registration(best_f1: float, rule_f1: float, champion_f1: float | None,
                        min_f1: float = 0.7, tolerance: float = 0.02):
    """回傳 (是否通過, 原因)。

    1. macro F1 不能低於絕對下限 min_f1
    2. 必須勝過規則式 baseline
    3. 不能比目前上線的模型（champion）差超過 tolerance
    注意：champion 與新模型用的資料不同（新模型多了新標註），所以這裡只做「不退步」的把關，
    不是嚴格的 A/B 比較；要更嚴謹請另外留一份固定的 hold-out 測試集。
    """
    if best_f1 < min_f1:
        return False, f"macro_f1 {best_f1:.3f} is below the minimum {min_f1:.2f}"
    if best_f1 <= rule_f1:
        return False, f"does not beat the rule-based baseline ({rule_f1:.3f})"
    if champion_f1 is not None and best_f1 < champion_f1 - tolerance:
        return False, f"worse than the current model ({champion_f1:.3f}) by more than {tolerance}"
    suffix = "" if champion_f1 is None else f" (current model {champion_f1:.3f})"
    return True, f"passed the gate{suffix}"

from pydantic import BaseModel, Field, field_validator

# 順序必須和訓練時的 FEATURES（dev_ 開頭那 6 欄）一致
DEV_NAMES = ["shoulder_tilt", "head_roll", "head_offset", "head_drop", "shrug", "lean"]
LABELS = ["good", "slouch", "lean_forward", "head_tilt", "shoulder_uneven"]


class FeaturePayload(BaseModel):
    """Edge 的 sender.py 送來的內容。只有 deviations 是必要的。"""
    ts: float | None = None
    deviations: dict[str, float]
    features: dict[str, float] | None = None
    issues: list[str] = Field(default_factory=list)

    @field_validator("deviations")
    @classmethod
    def _check_keys(cls, v):
        missing = [k for k in DEV_NAMES if k not in v]
        if missing:
            raise ValueError(f"deviations missing keys: {missing}")
        return v


class FeedbackPayload(BaseModel):
    """使用者更正：「剛剛這段時間，實際的姿勢是 label」。"""
    label: str
    seconds: float = Field(10.0, gt=0, le=120)   # 往回標記最近幾秒的資料

    @field_validator("label")
    @classmethod
    def _check_label(cls, v):
        if v not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        return v


class Prediction(BaseModel):
    label: str                      # 這一筆的原始預測
    stable_label: str               # 最近幾筆投票後的結果（減少閃爍）
    confidence: float
    probabilities: dict[str, float]
    model_source: str

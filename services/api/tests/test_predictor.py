import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.predictor import COLUMNS, ModelPredictor, RulePredictor
from app.schema import DEV_NAMES

GOOD = dict(shoulder_tilt=0.0, head_roll=0.0, head_offset=0.0, head_drop=0.0, shrug=0.0, lean=1.0)


def test_rule_predictor_each_class():
    p = RulePredictor()
    assert p.predict(GOOD)[0] == "good"
    assert p.predict({**GOOD, "head_drop": -0.3})[0] == "slouch"
    assert p.predict({**GOOD, "lean": 1.3})[0] == "lean_forward"
    assert p.predict({**GOOD, "head_roll": 12})[0] == "head_tilt"
    assert p.predict({**GOOD, "shoulder_tilt": 9})[0] == "shoulder_uneven"


def test_model_predictor_uses_correct_column_order():
    rng = np.random.default_rng(0)
    good = pd.DataFrame(rng.normal(0, 0.02, (100, 6)), columns=COLUMNS)
    good["dev_lean"] += 1.0
    slouch = good.copy()
    slouch["dev_head_drop"] -= 0.3
    X = pd.concat([good, slouch])
    y = ["good"] * 100 + ["slouch"] * 100
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    p = ModelPredictor(model, "test")
    assert [f"dev_{k}" for k in DEV_NAMES] == COLUMNS
    label, probs = p.predict({**GOOD, "head_drop": -0.3})
    assert label == "slouch"
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert p.predict(GOOD)[0] == "good"

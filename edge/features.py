"""從 MediaPipe Pose landmarks 計算正面視角的姿勢特徵。

所有長度類特徵都以「肩寬」正規化，因此不受離鏡頭遠近影響。
角度計算使用像素座標（乘上影像寬高），避免長寬比造成失真。
"""
import math

NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
L_SH, R_SH = 11, 12

FEATURE_NAMES = [
    "shoulder_tilt",  # 肩膀傾斜角（度），影像右側較低為正
    "head_roll",      # 頭部側傾角（度）
    "head_offset",    # 鼻子相對肩中點的水平偏移 / 肩寬
    "head_drop",      # 肩中點到鼻子的垂直距離 / 肩寬（駝背、低頭時變小）
    "shrug",          # 肩中點到耳朵的垂直距離 / 肩寬（聳肩時變小）
    "shoulder_w",     # 肩寬佔畫面寬度比例（靠近鏡頭時變大）
    "face_w",         # 雙眼距離佔畫面寬度比例（靠近鏡頭時變大）
]


def _pt(lm, i, w, h):
    return (lm[i].x * w, lm[i].y * h)


def _tilt_deg(a, b):
    """以 x 由小到大排序後，計算連線與水平線的夾角。"""
    if a[0] > b[0]:
        a, b = b, a
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def extract(landmarks, w: int, h: int, min_vis: float = 0.5):
    """回傳特徵 dict；若關鍵點不可見則回傳 None。"""
    for i in (NOSE, L_EYE, R_EYE, L_SH, R_SH):
        if landmarks[i].visibility < min_vis:
            return None

    nose = _pt(landmarks, NOSE, w, h)
    l_eye, r_eye = _pt(landmarks, L_EYE, w, h), _pt(landmarks, R_EYE, w, h)
    l_sh, r_sh = _pt(landmarks, L_SH, w, h), _pt(landmarks, R_SH, w, h)

    # 耳朵常被頭髮遮住，不可見時退而求其次用眼睛
    ears_ok = landmarks[L_EAR].visibility > 0.3 and landmarks[R_EAR].visibility > 0.3
    if ears_ok:
        l_h, r_h = _pt(landmarks, L_EAR, w, h), _pt(landmarks, R_EAR, w, h)
    else:
        l_h, r_h = l_eye, r_eye

    sh_w = math.dist(l_sh, r_sh)
    if sh_w < 1e-6:
        return None
    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    mid_head = ((l_h[0] + r_h[0]) / 2, (l_h[1] + r_h[1]) / 2)

    return {
        "shoulder_tilt": _tilt_deg(l_sh, r_sh),
        "head_roll": _tilt_deg(l_eye, r_eye),
        "head_offset": (nose[0] - mid_sh[0]) / sh_w,
        "head_drop": (mid_sh[1] - nose[1]) / sh_w,
        "shrug": (mid_sh[1] - mid_head[1]) / sh_w,
        "shoulder_w": sh_w / w,
        "face_w": math.dist(l_eye, r_eye) / w,
    }

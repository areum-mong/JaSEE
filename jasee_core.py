# -*- coding: utf-8 -*-
"""
jasee_core.py  ?? ?ì„¸?ˆë´ ?µì‹¬ ë¡œì§ (?µí•© ?•ì œ ë²„ì „)
?„ì¹˜: jasee_core.py

?¬í•¨ ?´ìš©
  - YOLOv8-pose 17ê°??¤í¬?¸íŠ¸ ê¸°ë°˜ ê°ë„ ê³„ì‚°
  - AttentionMLP ?ì„¸ ?ì •
  - ?˜ê²½ YOLO (chair_back / chair_seat / desk_surface / monitor)
  - ?¤ë²„?ˆì´ ê·¸ë¦¬ê¸?(posture / env)

ë³€ê²??´ë ¥ (ë¦¬íŒ©?°ë§)
  - ?‰ìƒ ?ìˆ˜ ?¨ì¼ ë¸”ë¡?¼ë¡œ ?µí•© (BGR)
    Â· COL_GOOD / COL_BAD / COL_ARROW / COL_NA
    Â· COLOR_GOOD / COLOR_BAD / COLOR_NA ??COL_* ë³„ì¹­?¼ë¡œ ?µì¼
  - _render_realtime_feedback_card ë¯¸ì‚¬???¨ìˆ˜ ?œê±° ??app_mobileë¡??´ê?
  - calc_gaze_angle ?¸ìëª??µì¼ (monitor_bbox)
  - ëª¨ë“  docstring ?œêµ­?´ë¡œ ?µì¼
"""

import math
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import pyttsx3
import cv2
from ultralytics import YOLO
from pathlib import Path

# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ê²½ë¡œ ?¤ì •
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
BASE_DIR      = Path(__file__).resolve().parent
POSE_MLP_PATH = BASE_DIR / "Yolo_pose" / "04_final_model" / "output" / "final_attention_mlp.pt"
ENV_YOLO_PATH = BASE_DIR / "Yolo_env" / "images_data" / "runs" / "posture_v1" / "weights" / "best.pt"

ENV_CLASSES = {0: "chair_back", 1: "chair_seat", 2: "desk_surface", 3: "monitor"}
ENV_COLORS  = {0: (0, 200, 255), 1: (0, 165, 255), 2: (255, 165, 0), 3: (255, 200, 0)}


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ì¸¡ì • ê¸°ì? (RULA + VDT ê³ ì‹œ ??020-17??
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
CRITERIA = {
    "CVA"        : (0,    20),    # ëª©êµ´ê³¡ê° 0~20Â° Good
    "TIA"        : (0,    20),    # ëª¸í†µêµ´ê³¡ê°?0~20Â° Good
    "knee_angle" : (85,  100),    # ë¬´ë¦ê°ë„ 85~100Â° Good
    "elbow_angle": (90,  120),    # ?”ê¿ˆì¹?ê°ë„ 90~120Â° Good
    "wrist_angle": (0,    15),    # ?ëª© ?¸ì°¨ Â±15Â° ?´ë‚´
    "gaze_angle" : (10,   15),    # ?œì„ ê°?10~15Â° ?˜ë°© Good
    "desk_diff"  : (0,  0.10),    # ?‘ì—…?€ ?’ì´ Â±10% ?´ë‚´
    "chair_gap"  : (0,  0.20),    # ?±ë°›??ê±°ë¦¬ ê³¨ë°˜?ˆë¹„ 20% ?´ë‚´
}


def is_good(key: str, value: float) -> bool:
    """CRITERIA ê¸°ì??¼ë¡œ ?´ë‹¹ ??ê°’ì´ ?•ìƒ ë²”ìœ„?¸ì? ë°˜í™˜?©ë‹ˆ??"""
    if key not in CRITERIA or value is None:
        return False
    lo, hi = CRITERIA[key]
    return lo <= value <= hi


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?¼ë“œë°?ë©”ì‹œì§€
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
FEEDBACK = {
    "CVA": {
        "no": "01", "label": "ëª©êµ´ê³¡ê°", "eng": "CVA", "cat": "posture",
        "range": "?•ìƒ 0Â°~20Â° Â· ?„í—˜ 20Â°ì´ˆê³¼",
        "good": "ë¨¸ë¦¬Â·ê²½ì¶” ?˜ì§ ?•ë ¬ ? ì?\nê²½ì¶” ë¶€??ìµœì†Œ???íƒœ",
        "bad":  "?„ë°©?ë??ì„¸(FHP) ?˜ì‹¬\nëª¨ë‹ˆ?°ë? ?ˆë†’?´ë¡œ ?¬ë¦¬?¸ìš”\n1?œê°„ë§ˆë‹¤ ëª??¤íŠ¸?ˆì¹­ ?œí–‰",
    },
    "TIA": {
        "no": "02", "label": "ëª¸í†µêµ´ê³¡ê°?, "eng": "TIA", "cat": "posture",
        "range": "?•ìƒ 0Â°~20Â° Â· ?„í—˜ 20Â°ì´ˆê³¼",
        "good": "ì²™ì¶” ?˜ì§ ?•ë ¬ ?‘í˜¸\n?”ì¶” ?•ë°• ìµœì†Œ???íƒœ",
        "bad":  "ê³¼ë„??ëª¸í†µ ?„êµ´ ê°ì?\n?±ë°›?´ì— ?ˆë¦¬ ?„ì „ ë°€ì°?n?˜ì ê¹Šìˆ™???‰ìœ¼?¸ìš”",
    },
    "ë¬´ë¦": {
        "no": "04", "label": "ë¬´ë¦ ê°ë„", "eng": "Knee", "cat": "posture",
        "range": "?•ìƒ 85Â°~100Â° Â· ?„í—˜ 85Â° ë¯¸ë§Œ ?ëŠ” 100Â° ì´ˆê³¼",
        "good": "?˜ì? ?ˆì•¡?œí™˜ ?í™œ\n?˜ì²´ ë¶€??ìµœì†Œ???íƒœ",
        "bad":  "ë¬´ë¦ ê°ë„ ê¸°ì? ?´íƒˆ\n?˜ì ?’ì´ ì¡°ì ˆ ?„ìš”\në°œë°›ì¹¨ë? ?¬ìš© ê¶Œì¥",
    },
    "?ëª©": {
        "no": "05", "label": "?ëª© ê°ë„", "eng": "Wrist", "cat": "posture",
        "range": "?•ìƒ Â±15Â° ?´ë‚´ ?ëª© ì¤‘ë¦½ ?ì„¸ ? ì?",
        "good": "?ëª© ì¤‘ë¦½ ?ì„¸ ? ì?\n?ëª© ?°ë„ ë¶€??ìµœì†Œ??,
        "bad":  "?ëª© ê³¼êµ´ê³?ê°ì?\n?ëª© ë°›ì¹¨?€ ?¤ì¹˜ ?„ìš”\n?¤ë³´????15cm ?•ë³´",
    },
    "?œì„ ê°?: {
        "no": "06", "label": "ëª¨ë‹ˆ???œì„ ê°?, "eng": "Gaze", "cat": "env",
        "range": "?•ìƒ ?˜ë°© 10Â°~15Â° Â· ?„í—˜ 10Â° ë¯¸ë§Œ ?ëŠ” 15Â° ì´ˆê³¼",
        "good": "?œì„ ê°?ê¸°ì? ì¶©ì¡±\nê²½ì¶” ë¶€??ìµœì†Œ??,
        "bad":  "?œì„ ê°?ê¸°ì? ?´íƒˆ\nëª¨ë‹ˆ???ë‹¨???ˆë†’?´ì— ë§ì¶”?¸ìš”\n?”ë©´ ê±°ë¦¬ 40cm ?´ìƒ ê¶Œì¥",
    },
    "ì±…ìƒ?’ì´": {
        "no": "07", "label": "?‘ì—…?€ ?’ì´", "eng": "Desk", "cat": "env",
        "range": "?•ìƒ Â±5% ?´ë‚´ Â· ?„í—˜ Â±5% ì´ˆê³¼",
        "good": "?‘ì—…?€Â·?”ê¿ˆì¹??•ë ¬ ?‘í˜¸\n?ì? ë¶€??ìµœì†Œ??,
        "bad":  "?‘ì—…?€ ?’ì´ ë¶ˆì¼ì¹?nì±…ìƒ ?’ì´ ?ëŠ” ?˜ì ?’ì´ ì¡°ì • ?„ìš”",
    },
    "?±ë°›??: {
        "no": "08", "label": "?˜ì ?±ë°›??, "eng": "Chair", "cat": "env",
        "range": "?•ìƒ 20% ?´ë‚´ Â· ?„í—˜ 20% ì´ˆê³¼",
        "good": "?±ë°›??ì§€ì§€ ì¶©ë¶„\n?”ì¶” ?ˆì •???•ë³´",
        "bad":  "?±ë°›??ì§€ì§€ ë¶€ì¡?n?˜ì ê¹Šìˆ™??ì°©ì„\n?ˆë¦¬ ?„ì „ ë°€ì°??„ìš”",
    },
}

INDICATOR_NAMES = {
    "CVA":      "CVA ëª©êµ´ê³¡ê°",
    "TIA":      "TIA ëª¸í†µêµ´ê³¡ê°?,
    "ë¬´ë¦":     "ë¬´ë¦ ê°ë„",
    "?ëª©":     "?ëª© ê°ë„",
    "?œì„ ê°?:   "ëª¨ë‹ˆ???œì„ ê°?,
    "ì±…ìƒ?’ì´": "?‘ì—…?€ ?’ì´",
    "?±ë°›??:   "?˜ì ?±ë°›??,
}

IND_UNITS = {
    "CVA": "Â°", "TIA": "Â°",
    "ë¬´ë¦": "Â°", "?ëª©": "Â°", "?œì„ ê°?: "Â°",
    "ì±…ìƒ?’ì´": "", "?±ë°›??: "",
}

DISPLAY_ORDER = ["CVA", "TIA", "ë¬´ë¦", "?ëª©", "?œì„ ê°?, "ì±…ìƒ?’ì´", "?±ë°›??]


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?¤ë²„?ˆì´ ?‰ìƒ (BGR) ???¨ì¼ ë¸”ë¡?¼ë¡œ ?µí•©
#
#  Â· COL_GOOD / COL_BAD / COL_ARROW / COL_NA  ???¤ì‹œê°??¤ë²„?ˆì´??(ì§„í•œ ê³„ì—´)
#  Â· COLOR_GOOD / COLOR_BAD / COLOR_NA         ???´ë?ì§€ ë¶„ì„ ?¤ë²„?ˆì´??(ë°ì? ê³„ì—´)
#    (app_mobile.py ?˜ìœ„?¸í™˜???„í•´ ë³„ì¹­?¼ë¡œ ? ì?)
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
COL_GOOD  = (100, 220, 100)   # ì´ˆë¡  ??STEP 1 ?¤ì‹œê°?
COL_BAD   = (50,   50, 220)   # ë¹¨ê°•  ??STEP 1 ?¤ì‹œê°?
COL_ARROW = (255, 100,   0)   # ?Œë‘ ?”ì‚´??
COL_NA    = (160, 160, 160)   # ?Œìƒ‰

COLOR_GOOD = (29,  158, 117)  # ë¯¼íŠ¸ ì´ˆë¡ ???´ë?ì§€ ë¶„ì„
COLOR_BAD  = (74,   50, 230)  # ?Œë‘-ë³´ë¼ ???´ë?ì§€ ë¶„ì„
COLOR_NA   = (180, 180, 180)  # ?Œìƒ‰

# STEP 1 ?¤ë²„?ˆì´ ê´€???¸ë±??(CVA/TIA ê´€?¨ë§Œ ?œì‹œ)
CVA_KP = [3, 4, 5, 6]        # ?¼ê?, ?¤ë¥¸ê·€, ?¼ì–´ê¹? ?¤ë¥¸?´ê¹¨
TIA_KP = [5, 6, 11, 12]      # ?´ê¹¨(??, ê³¨ë°˜(??


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?Œì„± ?ˆë‚´
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
_tts_lock = threading.Lock()


def speak(text: str):
    """TTS ?ˆë‚´ë¥?ë³„ë„ ?¤ë ˆ?œì—???¬ìƒ?©ë‹ˆ??"""
    def _run():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# AttentionMLP ëª¨ë¸ ?•ì˜
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
class AttentionMLP(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 256, num_classes: int = 1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=1, batch_first=True
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x):
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.net(attn_out.squeeze(1))


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ëª¨ë¸ ë¡œë“œ
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def load_models():
    """pose_yolo, mlp, env_yolo, device ë¥?ë°˜í™˜?©ë‹ˆ??"""
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    pose_yolo = YOLO("yolov8n-pose.pt")

    mlp = AttentionMLP()
    checkpoint = torch.load(POSE_MLP_PATH, map_location=device)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    mlp.load_state_dict(state_dict)
    mlp.eval().to(device)

    env_yolo = YOLO(str(ENV_YOLO_PATH))
    print(f"[ëª¨ë¸ ë¡œë“œ ?„ë£Œ] device={device}")
    return pose_yolo, mlp, env_yolo, device


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ê°ë„ ê³„ì‚° (YOLOv8-pose 17ê°??¸ë±??ê¸°ì?)
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def calc_angle_3pt(A, B, C) -> float:
    """???ì˜ ?´ê°??ê³„ì‚°?©ë‹ˆ??(Bê°€ ê¼?§“??."""
    v1 = np.array(A) - np.array(B)
    v2 = np.array(C) - np.array(B)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def calc_vertical_angle(p1, p2) -> float:
    """?˜ì§ì¶?ê¸°ì? ???ì˜ ê¸°ìš¸ê¸?ê°ë„ë¥?ë°˜í™˜?©ë‹ˆ??"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def calc_cva(lm) -> float:
    """CVA ëª©êµ´ê³¡ê° ??conf ?©ì‚°???’ì? ìª?ê·€?’ì–´ê¹?vs ?˜ì§."""
    r_conf = lm[4][2] + lm[6][2]
    l_conf = lm[3][2] + lm[5][2]
    if r_conf >= l_conf:
        ear      = np.array([lm[4][0], lm[4][1]])
        shoulder = np.array([lm[6][0], lm[6][1]])
    else:
        ear      = np.array([lm[3][0], lm[3][1]])
        shoulder = np.array([lm[5][0], lm[5][1]])
    return calc_vertical_angle(ear, shoulder)


def calc_tia(lm) -> float:
    """TIA ëª¸í†µêµ´ê³¡ê°????´ê¹¨ ì¤‘ì  ??ê³¨ë°˜ ì¤‘ì  vs ?˜ì§."""
    sh_mid  = np.array([(lm[5][0] + lm[6][0]) / 2,  (lm[5][1] + lm[6][1]) / 2])
    hip_mid = np.array([(lm[11][0] + lm[12][0]) / 2, (lm[11][1] + lm[12][1]) / 2])
    return calc_vertical_angle(sh_mid, hip_mid)


def calc_knee_angle(lm) -> float:
    """ë¬´ë¦ ê°ë„ ??conf ?’ì? ìª?ê³¨ë°˜-ë¬´ë¦-ë°œëª©."""
    r_conf = lm[12][2] + lm[14][2] + lm[16][2]
    l_conf = lm[11][2] + lm[13][2] + lm[15][2]
    if r_conf >= l_conf:
        return calc_angle_3pt(lm[12][:2], lm[14][:2], lm[16][:2])
    return calc_angle_3pt(lm[11][:2], lm[13][:2], lm[15][:2])


def calc_elbow_angle(lm) -> float | None:
    """?”ê¿ˆì¹?ê°ë„ ??conf ?’ì? ìª??´ê¹¨-?”ê¿ˆì¹??ëª©."""
    try:
        r_conf = lm[6][2] + lm[8][2] + lm[10][2]
        l_conf = lm[5][2] + lm[7][2] + lm[9][2]
        if r_conf >= l_conf:
            return calc_angle_3pt(lm[6][:2], lm[8][:2], lm[10][:2])
        return calc_angle_3pt(lm[5][:2], lm[7][:2], lm[9][:2])
    except Exception:
        return None


def calc_wrist_angle(lm) -> float | None:
    """?ëª© ?¸ì°¨ ???´ê¹¨-?”ê¿ˆì¹??ëª© ?´ê°?ì„œ 180Â° ì°¨ì´."""
    try:
        r_conf = lm[6][2] + lm[8][2] + lm[10][2]
        l_conf = lm[5][2] + lm[7][2] + lm[9][2]
        if r_conf >= l_conf:
            inner = calc_angle_3pt(lm[6][:2], lm[8][:2], lm[10][:2])
        else:
            inner = calc_angle_3pt(lm[5][:2], lm[7][:2], lm[9][:2])
        return float(abs(inner - 180.0))
    except Exception:
        return None


def calc_gaze_angle(lm, monitor_bbox) -> float | None:
    """ëª¨ë‹ˆ???œì„ ê°?????ì¤‘ì  ??ëª¨ë‹ˆ??ì¤‘ì‹¬."""
    if monitor_bbox is None:
        return None
    eye = np.array([(lm[1][0] + lm[2][0]) / 2, (lm[1][1] + lm[2][1]) / 2])
    mx  = (monitor_bbox[0] + monitor_bbox[2]) / 2
    my  = (monitor_bbox[1] + monitor_bbox[3]) / 2
    return float(np.degrees(np.arctan2(my - eye[1], mx - eye[0])))


def calc_desk_diff(lm, desk_bbox) -> float | None:
    """?‘ì—…?€ ?’ì´ ë¹„ìœ¨ ??conf ?’ì? ìª??”ê¿ˆì¹?vs ì±…ìƒ ?ë©´."""
    if desk_bbox is None:
        return None
    elbow_y   = lm[8][1] if lm[8][2] >= lm[7][2] else lm[7][1]
    desk_y    = desk_bbox[1]
    sh_mid_y  = (lm[5][1] + lm[6][1]) / 2
    hip_mid_y = (lm[11][1] + lm[12][1]) / 2
    ref       = abs(hip_mid_y - sh_mid_y)
    return float(abs(desk_y - elbow_y) / ref) if ref > 1e-4 else None


def calc_chair_gap(lm, chair_back_bbox) -> float | None:
    """?±ë°›??ê±°ë¦¬ ë¹„ìœ¨ ??ê³¨ë°˜ ì¤‘ì  vs chair_back ê°€ê¹Œìš´ ??"""
    if chair_back_bbox is None:
        return None
    hip_x      = (lm[11][0] + lm[12][0]) / 2
    back_left  = chair_back_bbox[0]
    back_right = chair_back_bbox[2]
    back_x     = back_left if abs(hip_x - back_left) < abs(hip_x - back_right) else back_right
    hip_w      = abs(lm[11][0] - lm[12][0])
    if hip_w < 5:
        hip_w = abs(lm[5][0] - lm[6][0])
    gap = abs(hip_x - back_x)
    return float(gap / hip_w) if hip_w > 1e-4 else None


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?ì„¸ MLP ?ì •
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def predict_posture(mlp, cva: float, tia: float, device: str) -> str:
    """AttentionMLP?ì„œ ê·œì¹™ ê¸°ë°˜?¼ë¡œ ?„í™˜. ?¬ìœ : validate_jasee.py ê²€ì¦?ê²°ê³¼ ê·œì¹™??ëª¨ë¸ë³´ë‹¤ ?•í™•???¬ëŒ ?¼ë²¨ 238??ê¸°ì? Accuracy 0.811 vs 0.578)"""
    cva_good = is_good("CVA", cva)
    tia_good = is_good("TIA", tia)
    
    result = "GOOD" if (cva_good and tia_good) else "BAD"
    print(f"[?ì •] CVA={cva:.1f}Â° ({'OK' if cva_good else 'NG'}), "
          f"TIA={tia:.1f}Â° ({'OK' if tia_good else 'NG'}) ??{result}")
    return result


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?ì„¸ ì¸¡ì • (?¤ì‹œê°??¹ìº  ?„ë ˆ??
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def run_posture(frame, pose_yolo, mlp, device, env_bboxes: dict) -> dict:
    """
    ?„ë ˆ??1?¥ì„ ë°›ì•„ ?ì„¸ ì¸¡ì • ê²°ê³¼ dictë¥?ë°˜í™˜?©ë‹ˆ??

    ë°˜í™˜ dict ??
      result        : 'GOOD' | 'BAD' | None
      metrics       : {CVA, TIA, knee_angle, elbow_angle, wrist_angle,
                       gaze_angle, desk_diff, chair_gap}
      keypoints     : {ê·€, ?´ê¹¨, ê³¨ë°˜, ë¬´ë¦, ë°œëª©, ?”ê¿ˆì¹? ?ëª©} ?½ì? ì¢Œí‘œ
      keypoints_raw : np.ndarray shape (17, 3) ???˜ê²½ ?„í„°ë§ìš©
      gate_pass     : bool (CVA + TIA ëª¨ë‘ GOOD)
    """
    out = {"result": None, "metrics": {}, "keypoints": {}, "gate_pass": False}
    pose_results = pose_yolo(frame, verbose=False)

    for r in pose_results:
        if r.keypoints is None:
            continue
        for kp in r.keypoints.data:
            if kp.shape[0] < 17:
                continue
            try:
                lm = kp.cpu().numpy()

                cva       = calc_cva(lm)
                tia       = calc_tia(lm)
                knee      = calc_knee_angle(lm)
                elbow     = calc_elbow_angle(lm)
                wrist     = calc_wrist_angle(lm)
                gaze      = calc_gaze_angle(lm, env_bboxes.get("monitor"))
                desk_diff = calc_desk_diff(lm, env_bboxes.get("desk_surface"))
                chair_gap = calc_chair_gap(lm, env_bboxes.get("chair_back"))

                out["metrics"] = {
                    "CVA": cva, "TIA": tia,
                    "knee_angle": knee, "elbow_angle": elbow,
                    "wrist_angle": wrist, "gaze_angle": gaze,
                    "desk_diff": desk_diff, "chair_gap": chair_gap,
                }
                out["result"]       = predict_posture(mlp, cva, tia, device)
                out["gate_pass"]    = is_good("CVA", cva) and is_good("TIA", tia)
                out["keypoints_raw"] = lm
                out["keypoints"] = {
                    "ê·€":    (int(lm[4][0]),  int(lm[4][1])),
                    "?´ê¹¨":  (int(lm[6][0]),  int(lm[6][1])),
                    "ê³¨ë°˜":  (int(lm[12][0]), int(lm[12][1])),
                    "ë¬´ë¦":  (int(lm[14][0]), int(lm[14][1])),
                    "ë°œëª©":  (int(lm[16][0]), int(lm[16][1])),
                    "?”ê¿ˆì¹?:(int(lm[8][0]),  int(lm[8][1])),
                    "?ëª©":  (int(lm[10][0]), int(lm[10][1])),
                }
            except Exception:
                pass
            break
        break

    return out


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?˜ê²½ ?¸ì‹
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def run_environment(frame, env_yolo, pose_keypoints=None) -> dict:
    """
    ?„ë ˆ?„ì—???‘ì—…?˜ê²½ ê°ì²´(?˜ì/ì±…ìƒ/ëª¨ë‹ˆ??ë¥??¸ì‹?©ë‹ˆ??

    ë°˜í™˜ dict ??
      detected : {label: conf}
      bboxes   : {label: [x1, y1, x2, y2]}

    ?„í„°ë§??„ëµ:
      - conf=0.5, iou=0.3 ?¼ë¡œ ì¤‘ë³µ ë°•ìŠ¤ ?œê±°
      - ?´ë˜?¤ë‹¹ conf ê°€???’ì? 1ê°œë§Œ ? ì?
      - pose_keypoints ?ˆìœ¼ë©?ê´€??ê·¼ì²˜ ê°ì²´ë§?? íƒ
          chair_back/chair_seat ??ê³¨ë°˜(11, 12) ê·¼ì²˜
          desk_surface          ???”ê¿ˆì¹?7~10) ê·¼ì²˜
          monitor               ???¼êµ´ ë°”ë¼ë³´ëŠ” ë°©í–¥ ?ìª½
    """
    detected, bboxes = {}, {}
    env_results = env_yolo(frame, verbose=False, conf=0.5, iou=0.3)

    # ?€?€ ?´ë˜?¤ë³„ ìµœê³  conf ?„ë³´ ?˜ì§‘ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    candidates = {}
    for r in env_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = ENV_CLASSES.get(cls, str(cls))
            if label not in candidates or conf > candidates[label]["conf"]:
                candidates[label] = {"conf": conf, "bbox": [x1, y1, x2, y2]}

    # pose_keypoints ?†ìœ¼ë©??„í„° ?†ì´ ë°˜í™˜
    if pose_keypoints is None:
        for label, v in candidates.items():
            detected[label] = v["conf"]
            bboxes[label]   = v["bbox"]
        return {"detected": detected, "bboxes": bboxes}

    kp  = pose_keypoints
    h, w = frame.shape[:2]

    def bbox_center(bbox):
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def pt_dist(p1, p2):
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    def kp_valid(idx):
        return kp[idx][2] > 0.3

    for label, v in candidates.items():
        bbox    = v["bbox"]
        cx, cy  = bbox_center(bbox)
        accept  = False

        if label in ("chair_back", "chair_seat"):
            hip_pts = [
                (kp[idx][0], kp[idx][1])
                for idx in [11, 12] if kp_valid(idx)
            ]
            if hip_pts:
                hip_cx = sum(p[0] for p in hip_pts) / len(hip_pts)
                hip_cy = sum(p[1] for p in hip_pts) / len(hip_pts)
                if pt_dist((cx, cy), (hip_cx, hip_cy)) < w * 0.6:
                    accept = True
            else:
                accept = True

        elif label == "desk_surface":
            # ?ëª©(9=?¼ì†ëª? 10=?¤ë¥¸?ëª©) ê·¼ì²˜ ì±…ìƒë§??¸ì‹
            best_wrist, best_conf = None, 0
            for idx in [9, 10]:
                if kp_valid(idx) and kp[idx][2] > best_conf:
                    best_conf  = kp[idx][2]
                    best_wrist = (kp[idx][0], kp[idx][1])
            if best_wrist:
                if pt_dist((cx, cy), best_wrist) < w * 0.6:
                    accept = True
            else:
                accept = True

        elif label == "monitor":
            # ?´ê¹¨ X ì¤‘ì  ???ëª© X ë°©í–¥?¼ë¡œ ëª¨ë‹ˆ?°ê? ?ˆëŠ”ì§€ ?ë³„
            # ì¸¡ë©´ ì´¬ì˜ ê¸°ì?: ?ëª©??ëª¨ë‹ˆ??ìª??´ê¹¨ ?ìª½)???ˆìŒ
            sh_pts = [kp[i][:2] for i in [5, 6] if kp_valid(i)]
            wrist_pts = [kp[i][:2] for i in [9, 10] if kp_valid(i)]
            if sh_pts and wrist_pts:
                sh_cx    = sum(p[0] for p in sh_pts) / len(sh_pts)
                wrist_cx = sum(p[0] for p in wrist_pts) / len(wrist_pts)
                # ?ëª©?’ì–´ê¹?ë°©í–¥ ë²¡í„° (ëª¨ë‹ˆ?°ê? ?ˆì–´????X ë°©í–¥)
                view_dx  = wrist_cx - sh_cx
                mon_dx   = cx - sh_cx
                # ëª¨ë‹ˆ?°ê? ?ëª© ë°©í–¥???ˆê³ , ?¬ëŒê³?ê°€??ê°€ê¹Œìš´ ê²?? íƒ
                if view_dx * mon_dx > 0:
                    accept = True
            else:
                accept = True

        if accept:
            detected[label] = v["conf"]
            bboxes[label]   = v["bbox"]

    return {"detected": detected, "bboxes": bboxes}


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?¤ë²„?ˆì´ ê·¸ë¦¬ê¸???STEP 1 ?ì„¸ ì¸¡ì • (?¤ì‹œê°?
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def draw_posture_overlay(frame: np.ndarray, lm: np.ndarray, metrics: dict) -> np.ndarray:
    """
    CVA/TIA ê´€??ê´€?ˆë§Œ ?¤ì‹œê°??¤ë²„?ˆì´?©ë‹ˆ??
      - GOOD: ì´ˆë¡(COL_GOOD) ????
      - BAD:  ë¹¨ê°•(COL_BAD) ????+ ?Œë?(COL_ARROW) ?”ì‚´??

    Args:
        frame   : BGR ?„ë ˆ??
        lm      : shape (17, 3) numpy array  (x, y, conf)
        metrics : {"CVA": float|None, "TIA": float|None, ...}

    Returns:
        ?¤ë²„?ˆì´ê°€ ?ìš©??BGR ?„ë ˆ??ë³µì‚¬ë³?
    """
    out = frame.copy()

    cva_raw = metrics.get("CVA")
    tia_raw = metrics.get("TIA")
    cva_ok  = is_good("CVA", cva_raw) if cva_raw is not None else None
    tia_ok  = is_good("TIA", tia_raw) if tia_raw is not None else None

    def kp_color(ok):
        if ok is None:
            return COL_NA
        return COL_GOOD if ok else COL_BAD

    cva_col = kp_color(cva_ok)
    tia_col = kp_color(tia_ok)

    KP_COL = {
        3: cva_col, 4: cva_col,      # ê·€
        5: tia_col, 6: tia_col,      # ?´ê¹¨
        11: tia_col, 12: tia_col,    # ê³¨ë°˜
    }

    LINES = [
        (3, 5, cva_col), (4, 6, cva_col),          # ê·€-?´ê¹¨ (CVA)
        (5, 6, tia_col),                             # ?´ê¹¨-?´ê¹¨
        (5, 11, tia_col), (6, 12, tia_col),          # ?´ê¹¨-ê³¨ë°˜ (TIA)
        (11, 12, tia_col),                           # ê³¨ë°˜-ê³¨ë°˜
    ]
    for a, b, col in LINES:
        if lm[a][2] > 0.3 and lm[b][2] > 0.3:
            pa = (int(lm[a][0]), int(lm[a][1]))
            pb = (int(lm[b][0]), int(lm[b][1]))
            cv2.line(out, pa, pb, (0, 0, 0), 1)
            cv2.line(out, pa, pb, col, 1)

    for idx, col in KP_COL.items():
        if lm[idx][2] < 0.3:
            continue
        px, py = int(lm[idx][0]), int(lm[idx][1])
        if col == COL_BAD:
            cv2.circle(out, (px, py), 1, col, -1)
            cv2.circle(out, (px, py), 2, (255, 255, 255), 1)
            cv2.circle(out, (px, py), 3, col, 1)
        else:
            cv2.circle(out, (px, py), 1, col, -1)
            cv2.circle(out, (px, py), 2, (255, 255, 255), 1)

    # CVA BAD ??ê·€?ì„œ ?¤ìª½(?„ë°©) ?”ì‚´??
    if cva_ok is False:
        ear_idx = 4 if lm[4][2] >= lm[3][2] else 3
        if lm[ear_idx][2] > 0.3:
            ep  = (int(lm[ear_idx][0]), int(lm[ear_idx][1]))
            tgt = (ep[0] + 60, ep[1] - 15)
            cv2.arrowedLine(out, ep, tgt, COL_ARROW, 1, tipLength=0.35)
            cv2.circle(out, tgt, 1, COL_ARROW, -1)

    # TIA BAD ???´ê¹¨?ì„œ ?„ìª½ ?”ì‚´??
    if tia_ok is False:
        sh_idx = 6 if lm[6][2] >= lm[5][2] else 5
        if lm[sh_idx][2] > 0.3:
            sp  = (int(lm[sh_idx][0]), int(lm[sh_idx][1]))
            tgt = (sp[0], sp[1] - 70)
            cv2.arrowedLine(out, sp, tgt, COL_ARROW, 1, tipLength=0.35)
            cv2.circle(out, tgt, 1, COL_ARROW, -1)

    return out


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?¤ë²„?ˆì´ ê·¸ë¦¬ê¸???STEP 2 ?˜ê²½ ì¸¡ì • (?¤ì‹œê°?
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def draw_env_overlay(
    frame: np.ndarray,
    bboxes: dict,
    detected: dict,
    lm: np.ndarray = None,
    metrics: dict = None,
) -> np.ndarray:
    """
    ê°ì????˜ê²½ ê°ì²´??êµµì? ë°•ìŠ¤ + ?œê? ?¼ë²¨???œì‹œ?©ë‹ˆ??
    ?ì„¸ GOOD ?íƒœ???Œë§Œ ?Œë? ì¡°ì • ?”ì‚´?œë? ì¶”ê?ë¡??œì‹œ?©ë‹ˆ??
    (BAD ?íƒœ?ì„œ???˜ê²½ ë°•ìŠ¤ë§??œì‹œ?˜ê³  ?”ì‚´?œëŠ” ?ëµ)

    Args:
        frame    : BGR ?„ë ˆ??
        bboxes   : {label: [x1,y1,x2,y2]}
        detected : {label: conf}
        lm       : shape (17, 3) numpy array (? íƒ)
        metrics  : {"CVA": float, "TIA": float, ...} (? íƒ)
    """
    out  = frame.copy()
    h, w = out.shape[:2]

    LABEL_KR = {
        "chair_back":   "?±ë°›??,
        "chair_seat":   "?˜ì?œíŠ¸",
        "desk_surface": "ì±…ìƒ",
        "monitor":      "ëª¨ë‹ˆ??,
    }
    BOX_COLORS = {
        "chair_back":   (255, 180,   0),
        "chair_seat":   (255, 140,   0),
        "desk_surface": (0,   180, 255),
        "monitor":      (180, 255,   0),
    }

    posture_good = False
    if metrics:
        cva_ok = is_good("CVA", metrics.get("CVA")) if metrics.get("CVA") is not None else False
        tia_ok = is_good("TIA", metrics.get("TIA")) if metrics.get("TIA") is not None else False
        posture_good = cva_ok and tia_ok

    # ?˜ê²½ ë°•ìŠ¤ + ?¼ë²¨
    for label, bbox in bboxes.items():
        x1, y1, x2, y2 = bbox
        col = BOX_COLORS.get(label, (200, 200, 200))
        lbl = LABEL_KR.get(label, label)

        cv2.rectangle(out, (x1, y1), (x2, y2), col, 5)

        font_scale = 1.1
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(out, (x1, y1 - th - 20), (x1 + tw + 12, y1), col, -1)
        cv2.putText(out, lbl, (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    # ì²´í¬ë¦¬ìŠ¤??(ì¢Œì¸¡)
    required = ["chair_back", "desk_surface", "monitor"]
    for i, item in enumerate(required):
        ok  = item in detected
        col = (100, 220, 100) if ok else (80, 80, 80)
        lbl = LABEL_KR.get(item, item)
        mark = "v" if ok else "o"
        cv2.putText(out, f"{mark} {lbl}",
                    (15, 80 + i * 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)

    # ?ì„¸ GOOD ???Œë§Œ ì¡°ì • ?”ì‚´???œì‹œ
    if posture_good and lm is not None:
        # ëª¨ë‹ˆ?? ?ˆë†’????ëª¨ë‹ˆ??ì¤‘ì‹¬
        if "monitor" in bboxes and lm[1][2] > 0.3 and lm[2][2] > 0.3:
            eye_y  = int((lm[1][1] + lm[2][1]) / 2)
            eye_x  = int((lm[1][0] + lm[2][0]) / 2)
            mon    = bboxes["monitor"]
            mon_cy = (mon[1] + mon[3]) // 2
            if abs(eye_y - mon_cy) > 30:
                cv2.arrowedLine(out, (eye_x, eye_y), (eye_x, mon_cy),
                                COL_ARROW, 1, tipLength=0.2)
                cv2.circle(out, (eye_x, mon_cy), 1, COL_ARROW, -1)

        # ì±…ìƒ: ?”ê¿ˆì¹??’ì´ ??ì±…ìƒë©?
        if "desk_surface" in bboxes and lm is not None:
            el_idx = 8 if lm[8][2] >= lm[7][2] else 7
            if lm[el_idx][2] > 0.3:
                ep     = (int(lm[el_idx][0]), int(lm[el_idx][1]))
                desk_y = bboxes["desk_surface"][1]
                if abs(ep[1] - desk_y) > 20:
                    tgt = (ep[0], desk_y)
                    cv2.arrowedLine(out, ep, tgt, COL_ARROW, 1, tipLength=0.25)
                    cv2.circle(out, tgt, 1, COL_ARROW, -1)

    return out





# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?œê? ?ìŠ¤???¤ë²„?ˆì´ (PIL ê²½ìœ )
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
from PIL import Image as _PILImage, ImageDraw as _ImageDraw, ImageFont as _ImageFont

def cv2_put_korean(img: np.ndarray, text: str, pos: tuple,
                   font_size: int = 22, color: tuple = (255, 100, 0)) -> np.ndarray:
    """OpenCV BGR ?´ë?ì§€???œê? ?ìŠ¤?¸ë? PIL ê²½ìœ ë¡?ê·¸ë¦½?ˆë‹¤."""
    try:
        img_pil = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw    = _ImageDraw.Draw(img_pil)
        font    = None
        import config`n        for fp in [config.korean_font()]:
            try:
                font = _ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = _ImageFont.load_default()
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return img


# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
# ?´ë?ì§€ ë¶„ì„ ?¤ë²„?ˆì´ ??TAP 2 / ?¤ì‹œê°?ê²°ê³¼ ê³µí†µ
# ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def draw_image_overlay(
    frame: np.ndarray,
    kp: dict,
    all_kr: dict,
    bboxes: dict,
    detected: dict,
    metrics: dict,
) -> np.ndarray:
    """
    ?´ë?ì§€ ë¶„ì„ ê²°ê³¼(7ê°?ì§€???„ì²´)ë¥?BGR ?„ë ˆ?„ì— ?¤ë²„?ˆì´?©ë‹ˆ??

    Args:
        frame    : BGR numpy ë°°ì—´
        kp       : {ê´€?ˆëª…: (x,y)} ??run_posture??keypoints ë°˜í™˜ê°?
        all_kr   : {?œê??? (val_str, is_ok, raw)} ??posture+env ?µí•©
        bboxes   : {label: [x1,y1,x2,y2]} ???˜ê²½ ê°ì²´ bbox
        detected : {label: conf} ???˜ê²½ ê°ì²´ ? ë¢°??
        metrics  : {?ì–´?? float} ??ê°ë„/ê±°ë¦¬ ?ì‹œê°?(?°ì¸¡ ?˜ì¹˜ ?œì‹œ??

    Returns:
        ?¤ë²„?ˆì´ê°€ ?ìš©??BGR ë°°ì—´
    """
    ARROW_COL = COL_ARROW  # (255, 100, 0) ???Œë‘ ?”ì‚´??

    def ind_color(kr_key):
        if kr_key in all_kr:
            _, is_ok, raw = all_kr[kr_key]
            if raw is None:
                return COLOR_NA
            return COLOR_GOOD if is_ok else COLOR_BAD
        return COLOR_NA

    cva_col   = ind_color("CVA")
    tia_col   = ind_color("TIA")
    knee_col  = ind_color("ë¬´ë¦")
    wrist_col = ind_color("?ëª©")
    chair_col = ind_color("?±ë°›??)

    KP_COL = {
        "ê·€":    cva_col,
        "?´ê¹¨":  tia_col,
        "ê³¨ë°˜":  tia_col,
        "ë¬´ë¦":  knee_col,
        "ë°œëª©":  knee_col,
        "?”ê¿ˆì¹?: wrist_col,
        "?ëª©":  wrist_col,
    }
    LINE_COL = {
        ("ê·€",  "?´ê¹¨"):    cva_col,
        ("?´ê¹¨", "ê³¨ë°˜"):   tia_col,
        ("ê³¨ë°˜", "ë¬´ë¦"):   knee_col,
        ("ë¬´ë¦", "ë°œëª©"):   knee_col,
        ("?´ê¹¨", "?”ê¿ˆì¹?): wrist_col,
        ("?”ê¿ˆì¹?, "?ëª©"): wrist_col,
    }

    out = frame.copy()
    h, w = out.shape[:2]

    if kp:
        # ?€?€ ?°ê²°???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
        for (a, b), lc in LINE_COL.items():
            if a in kp and b in kp:
                cv2.line(out, kp[a], kp[b], (0, 0, 0), 20)
                cv2.line(out, kp[a], kp[b], lc, 12)

        # ?€?€ ê´€?????€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
        for name, pt in kp.items():
            c = KP_COL.get(name, COLOR_NA)
            if c == COLOR_BAD:
                cv2.circle(out, pt, 26, c, -1)
                cv2.circle(out, pt, 33, (255, 255, 255), 5)
                cv2.circle(out, pt, 38, c, 4)
            elif c == COLOR_GOOD:
                cv2.circle(out, pt, 22, c, -1)
                cv2.circle(out, pt, 28, (255, 255, 255), 4)
            else:
                cv2.circle(out, pt, 16, c, -1)
                cv2.circle(out, pt, 21, (255, 255, 255), 3)

        # ?€?€ CVA BAD: ê·€ ?????”ì‚´???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
        if cva_col == COLOR_BAD and "ê·€" in kp and "?´ê¹¨" in kp:
            ear    = kp["ê·€"]
            target = (ear[0] + 55, ear[1] - 15)
            cv2.arrowedLine(out, ear, target, ARROW_COL, 10, tipLength=0.4)
            out = cv2_put_korean(out, "ë¨¸ë¦¬ë¥??¤ë¡œ", (ear[0] + 5, ear[1] - 28), 20, ARROW_COL)

        # ?€?€ TIA BAD: ?´ê¹¨ ?????”ì‚´???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
        if tia_col == COLOR_BAD and "?´ê¹¨" in kp:
            sh     = kp["?´ê¹¨"]
            target = (sh[0], sh[1] - 65)
            cv2.arrowedLine(out, sh, target, ARROW_COL, 10, tipLength=0.4)
            out = cv2_put_korean(out, "ëª¸í†µ ?¸ìš°ê¸?, (sh[0] + 8, sh[1] - 76), 20, ARROW_COL)

        # ?€?€ ë¬´ë¦ BAD: ê°ë„ ê¸°ë°˜ ë°©í–¥ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
        if knee_col == COLOR_BAD and "ë¬´ë¦" in kp:
            knee  = kp["ë¬´ë¦"]
            k_raw = (all_kr.get("ë¬´ë¦") or (None, None, None))[2]
            if k_raw is not None:
                if float(k_raw) < 85:
                    target = (knee[0], knee[1] + 70)
                    cv2.arrowedLine(out, knee, target, ARROW_COL, 10, tipLength=0.4)
                    out = cv2_put_korean(out, "?˜ì ??¶”ê¸?, (knee[0] + 8, knee[1] + 75), 20, ARROW_COL)
                else:
                    target = (knee[0], knee[1] - 70)
                    cv2.arrowedLine(out, knee, target, ARROW_COL, 10, tipLength=0.4)
                    out = cv2_put_korean(out, "?˜ì ?¬ë¦¬ê¸?, (knee[0] + 8, knee[1] - 80), 20, ARROW_COL)

        # ?€?€ ?ëª© BAD: ?˜í‰ ?”ì‚´???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
        if wrist_col == COLOR_BAD and "?ëª©" in kp:
            wrist  = kp["?ëª©"]
            target = (wrist[0] - 55, wrist[1])
            cv2.arrowedLine(out, wrist, target, ARROW_COL, 10, tipLength=0.4)
            out = cv2_put_korean(out, "?ëª© ì¤‘ë¦½", (wrist[0] - 90, wrist[1] - 14), 20, ARROW_COL)

    # ?€?€ ?±ë°›???´ê²©ê±°ë¦¬ ?œê°???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    chair_bbox = bboxes.get("chair_back")
    if chair_bbox and "ê³¨ë°˜" in kp:
        hip_pt  = kp["ê³¨ë°˜"]
        back_x  = chair_bbox[2]
        back_y1 = chair_bbox[1]
        gap_color = COLOR_GOOD if chair_col == COLOR_GOOD else COLOR_BAD
        cv2.line(out, (back_x, hip_pt[1]), hip_pt, gap_color, 4)
        mid_x = (back_x + hip_pt[0]) // 2
        cv2.circle(out, (mid_x, hip_pt[1]), 6, gap_color, -1)
        cv2.rectangle(out,
                      (chair_bbox[0], chair_bbox[1]),
                      (chair_bbox[2], chair_bbox[3]),
                      gap_color, 3)
        if chair_col == COLOR_BAD:
            out = cv2_put_korean(out, "??ë°€ì°??„ìš”", (back_x + 5, back_y1 - 10), 18, ARROW_COL)

    # ?€?€ ?°ì¸¡ ?ë‹¨: ì§€???˜ì¹˜ ?œì‹œ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    y_off = 55
    for key in DISPLAY_ORDER:
        raw = metrics.get(key)
        if raw is None:
            continue
        g    = is_good(key, raw)
        c    = COLOR_GOOD if g else COLOR_BAD
        unit = IND_UNITS.get(key, "")
        cv2.putText(out,
                    f"{INDICATOR_NAMES.get(key, key)}: {raw:.1f}{unit}",
                    (w - 250, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        y_off += 22

    # ?€?€ ?˜ê²½ ê°ì²´ bbox ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
    for label, bbox in bboxes.items():
        x1, y1, x2, y2 = bbox
        cls_keys = [k for k, v in ENV_CLASSES.items() if v == label]
        c        = ENV_COLORS.get(cls_keys[0], (200, 200, 200)) if cls_keys else (200, 200, 200)
        conf     = detected.get(label, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"{label} {conf:.0%}",
                    (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    return out



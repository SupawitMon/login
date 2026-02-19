import os
import time
import uuid

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import streamlit as st
import requests


# ==========================================
# CONFIG (คงเดิม)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model.pth"
HF_MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth?download=true"

UPLOAD_FOLDER = "static/uploads"

CRACK_THRESHOLD = 0.58
HIT_THRESHOLD = 0.48
HIT_K = 2

USE_MULTI_CROP = True
CROP_RATIO = 0.75
USE_9_CROP = True

STONE_LAP_MIN = 90.0
STONE_EDGE_MIN = 0.015

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# Download model
# ==========================================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    st.info("Downloading model from Hugging Face...")
    r = requests.get(HF_MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


# ==========================================
# Load model
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model_and_meta():
    ensure_model()

    try:
        if hasattr(torch, "serialization"):
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    state = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    class_to_idx = ckpt.get("class_to_idx")
    crack_idx = class_to_idx["Crack"]
    no_crack_idx = class_to_idx["No Crack"]

    IMG_SIZE = int(ckpt.get("img_size", 300))
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, crack_idx, no_crack_idx, transform


# ==========================================
# CV Stone Gate
# ==========================================
def is_stone_cv(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return lap >= STONE_LAP_MIN and edge_density >= STONE_EDGE_MIN, lap, edge_density


def stone_confidence(lap, edge):
    lap_score = max(0, min(1, (lap - STONE_LAP_MIN) / (STONE_LAP_MIN * 0.8)))
    edge_score = max(0, min(1, (edge - STONE_EDGE_MIN) / STONE_EDGE_MIN))
    return round((0.6 * lap_score + 0.4 * edge_score) * 100, 2)


# ==========================================
# AI Predict
# ==========================================
def _predict(pil, model, tfm, ci, ni):
    x = tfm(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    return float(p[ci]), float(p[ni])


def predict_image_ai(pil, model, tfm, ci, ni):
    if not USE_MULTI_CROP:
        c, n = _predict(pil, model, tfm, ci, ni)
        return c, n, [c]

    W, H = pil.size
    s = int(min(W, H) * CROP_RATIO)
    s = max(32, s)

    def box(x, y): return (x, y, x + s, y + s)

    boxes = [
        box(0, 0), box(W - s, 0),
        box(0, H - s), box(W - s, H - s),
        box((W - s)//2, (H - s)//2)
    ]

    if USE_9_CROP:
        boxes += [
            box((W - s)//2, 0), box((W - s)//2, H - s),
            box(0, (H - s)//2), box(W - s, (H - s)//2)
        ]

    cps = []
    for b in boxes:
        c, _ = _predict(pil.crop(b), model, tfm, ci, ni)
        cps.append(c)

    return max(cps), 1 - max(cps), cps


def decide_crack(mx, cps):
    hits = sum(p >= HIT_THRESHOLD for p in cps)
    return mx >= CRACK_THRESHOLD or hits >= HIT_K, hits


# ==========================================
# UI
# ==========================================
st.set_page_config("Stone AI Inspection", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Inter:wght@400;500;600&display=swap');

:root{
--bg:#0b1423;
--card:rgba(255,255,255,0.06);
--accent1:#00bfff;
--accent2:#00ffcc;
--success:#00e676;
--danger:#ff5252;
--warning:#ff9800;
}

html, body, [class*="css"]{
background:var(--bg);
color:#e5e7eb;
font-family:'Inter',sans-serif;
}

.stApp::before{
content:"";
position:fixed;
inset:0;
background-image:
linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
background-size:40px 40px;
animation:grid 25s linear infinite;
z-index:-1;
}
@keyframes grid{
from{background-position:0 0;}
to{background-position:160px 160px;}
}

.bigTitle{
font-family:'Orbitron',sans-serif;
font-size:42px;
text-align:center;
}
.ai{
background:linear-gradient(270deg,var(--accent1),var(--accent2),#8b5cf6,var(--accent1));
background-size:600% 600%;
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
animation:flow 6s ease infinite;
}
@keyframes flow{
0%{background-position:0% 50%}
50%{background-position:100% 50%}
100%{background-position:0% 50%}
}

.badge{
padding:18px 36px;
border-radius:999px;
font-weight:700;
display:inline-block;
margin-top:20px;
}
.success{color:var(--success);border:1px solid var(--success);}
.danger{color:var(--danger);border:1px solid var(--danger);}
.warning{color:var(--warning);border:1px solid var(--warning);}

.card{
background:var(--card);
border-radius:18px;
padding:20px;
text-align:center;
border:1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

model, crack_idx, no_crack_idx, transform = load_model_and_meta()

st.markdown("""
<div class="bigTitle">
Stone Defect Detection <span class="ai">AI</span>
</div>
<p style="text-align:center;opacity:.8">
ระบบตรวจสอบคุณภาพของหินด้วยเทคโนโลยี AI Vision
</p>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("อัปโหลดรูปหิน", type=list(ALLOWED_EXT))
run = st.button("ตรวจสอบคุณภาพ")

if run and uploaded:
    path = f"{UPLOAD_FOLDER}/{uuid.uuid4().hex}.jpg"
    open(path, "wb").write(uploaded.getvalue())

    bgr = cv2.imread(path)
    ok, lap, edge = is_stone_cv(bgr)

    if not ok:
        conf = stone_confidence(lap, edge)
        st.markdown(f'<div class="badge warning">❌ ไม่ใช่หิน<br>{conf:.2f}%</div>', unsafe_allow_html=True)
    else:
        pil = Image.open(path).convert("RGB")
        mx, _, cps = predict_image_ai(pil, model, transform, crack_idx, no_crack_idx)
        crack, hits = decide_crack(mx, cps)
        conf = mx * 100

        cls = "danger" if crack else "success"
        txt = "❌ พบรอยแตก" if crack else "✅ ไม่พบรอยแตก"

        st.markdown(f'<div class="badge {cls}">{txt}<br>{conf:.2f}%</div>', unsafe_allow_html=True)
        st.progress(conf / 100)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="card"><small>Crack Count</small><h2>{1 if crack else 0}</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card"><small>Processing Time</small><h2>{0.0:.3f}s</h2></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card"><small>AI Confidence</small><h2>{conf:.2f}%</h2></div>', unsafe_allow_html=True)

        st.image(path, caption="Result", use_container_width=True)

st.caption("© 2026 Stone AI Inspection | Advanced Vision Technology")

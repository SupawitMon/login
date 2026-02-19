# ============================================================
# Stone Crack AI Inspection - FULL SYSTEM + ULTRA UI
# ============================================================

# ===============================
# [A] IMPORTS
# ===============================
import os
import time
import uuid
import requests

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import streamlit as st

# ===============================
# [B] CONFIG (SYSTEM - LOCKED)
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model.pth"
HF_MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- Decision thresholds (DO NOT TOUCH) ----
CRACK_THRESHOLD = 0.58
HIT_THRESHOLD = 0.48
HIT_K = 2

USE_MULTI_CROP = True
USE_9_CROP = True
CROP_RATIO = 0.75

STONE_LAP_MIN = 90.0
STONE_EDGE_MIN = 0.015

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# ===============================
# [C] MODEL DOWNLOAD
# ===============================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    with st.status("📥 Downloading model from HuggingFace...", expanded=True):
        r = requests.get(HF_MODEL_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded")

# ===============================
# [D] LOAD MODEL (SAFE)
# ===============================
@st.cache_resource(show_spinner=False)
def load_model():
    ensure_model()

    try:
        if hasattr(torch.serialization, "add_safe_globals"):
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

    class_to_idx = ckpt.get("class_to_idx", {"Crack": 0, "No Crack": 1})
    crack_idx = class_to_idx["Crack"]
    no_crack_idx = class_to_idx["No Crack"]

    IMG_SIZE = int(ckpt.get("img_size", 300)) if isinstance(ckpt, dict) else 300

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, transform, crack_idx, no_crack_idx

# ===============================
# [E] CV STONE GATE
# ===============================
def is_stone_cv(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    ok = lap >= STONE_LAP_MIN and edge_density >= STONE_EDGE_MIN
    return ok, lap, edge_density

def stone_confidence(lap, edge):
    lap_score = min(1, max(0, (lap - STONE_LAP_MIN) / (STONE_LAP_MIN * 0.8)))
    edge_score = min(1, max(0, (edge - STONE_EDGE_MIN) / STONE_EDGE_MIN))
    return round((0.6 * lap_score + 0.4 * edge_score) * 100, 2)

# ===============================
# [F] AI PREDICT
# ===============================
def _predict(img, model, tfm, crack_idx, no_idx):
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    return float(p[crack_idx]), float(p[no_idx])

def predict_multi(img, model, tfm, crack_idx, no_idx):
    if not USE_MULTI_CROP:
        c, n = _predict(img, model, tfm, crack_idx, no_idx)
        return c, n, [c]

    W, H = img.size
    cs = int(min(W, H) * CROP_RATIO)
    cs = max(cs, 32)

    boxes = [
        (0, 0), (W - cs, 0), (0, H - cs), (W - cs, H - cs),
        ((W - cs)//2, (H - cs)//2)
    ]
    if USE_9_CROP:
        boxes += [
            ((W - cs)//2, 0), ((W - cs)//2, H - cs),
            (0, (H - cs)//2), (W - cs, (H - cs)//2)
        ]

    probs = []
    for x, y in boxes:
        crop = img.crop((x, y, x + cs, y + cs))
        c, _ = _predict(crop, model, tfm, crack_idx, no_idx)
        probs.append(c)

    return max(probs), 1 - max(probs), probs

def decide(crack_max, probs):
    hits = sum(p >= HIT_THRESHOLD for p in probs)
    return crack_max >= CRACK_THRESHOLD or hits >= HIT_K, hits

# ===============================
# [G] STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Stone Crack AI Inspection",
    page_icon="🪨",
    layout="wide"
)

# ===============================
# [H] ULTRA PREMIUM CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;500;600&display=swap');
html, body { background:#0b1423; color:white; font-family:Inter; }
.title {
    font-family:Orbitron;
    font-size:48px;
    text-align:center;
    background:linear-gradient(90deg,#00f5ff,#00ffcc,#8b5cf6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.glass {
    background:rgba(255,255,255,0.05);
    border-radius:24px;
    padding:32px;
    border:1px solid rgba(255,255,255,0.1);
}
.ring {
    width:180px;height:180px;border-radius:50%;
    margin:auto;
    display:flex;align-items:center;justify-content:center;
    font-size:22px;font-weight:700;
    background:
      radial-gradient(circle,#0b1423 45%,transparent 46%),
      conic-gradient(#00ffcc VAR%,#1e293b VAR%);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# [I] LOAD MODEL
# ===============================
with st.spinner("Loading AI model..."):
    model, transform, crack_idx, no_crack_idx = load_model()

# ===============================
# [J] UI
# ===============================
st.markdown('<div class="title">Stone Crack AI Inspection</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Stone Image", type=["jpg","png","jpeg","webp","bmp","gif"])
    run = st.button("🔍 Scan Stone", use_container_width=True)

    if run and uploaded:
        ext = os.path.splitext(uploaded.name)[1].lower()
        if ext == ".gif":
            st.error("❌ ยังไม่รองรับไฟล์ GIF")
        else:
            path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
            with open(path, "wb") as f:
                f.write(uploaded.getvalue())

            img_bgr = cv2.imread(path)
            ok, lap, edge = is_stone_cv(img_bgr)

            col1, col2 = st.columns([1,1])
            col1.image(path, use_container_width=True)

            if not ok:
                conf = stone_confidence(lap, edge)
                col2.warning("❌ ไม่ใช่หิน")
                col2.progress(conf/100)
            else:
                pil = Image.open(path).convert("RGB")
                crack_max, _, probs = predict_multi(
                    pil, model, transform, crack_idx, no_crack_idx
                )
                crack, hits = decide(crack_max, probs)
                conf = round((crack_max if crack else 1-crack_max)*100, 2)

                ring = f"""
                <div class="ring" style="--percent:{int(conf)};background:
                radial-gradient(circle,#0b1423 45%,transparent 46%),
                conic-gradient(#00ffcc {conf}%,#1e293b {conf}%);">
                {conf}%
                </div>
                """
                col2.markdown(ring, unsafe_allow_html=True)
                if crack:
                    col2.error("❌ พบรอยแตก")
                else:
                    col2.success("✅ ไม่พบรอยแตก")

    st.markdown('</div>', unsafe_allow_html=True)

st.caption("© 2026 Stone AI Inspection | Production AI System")

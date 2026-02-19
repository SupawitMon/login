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
# CONFIG (AI LOGIC = คงเดิม)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model.pth"
HF_MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth?download=true"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CRACK_THRESHOLD = 0.58
HIT_THRESHOLD = 0.48
HIT_K = 2

USE_MULTI_CROP = True
CROP_RATIO = 0.75
USE_9_CROP = True

STONE_LAP_MIN = 90.0
STONE_EDGE_MIN = 0.015

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# ==========================================
# MODEL LOADING
# ==========================================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    r = requests.get(HF_MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model():
    ensure_model()
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    class_to_idx = ckpt["class_to_idx"]
    crack_idx = class_to_idx["Crack"]
    no_crack_idx = class_to_idx["No Crack"]

    IMG_SIZE = ckpt.get("img_size", 300)
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
# CV GATE
# ==========================================
def is_stone_cv(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (bgr.shape[0] * bgr.shape[1])
    return lap >= STONE_LAP_MIN and edge_density >= STONE_EDGE_MIN, lap, edge_density

# ==========================================
# AI PREDICT
# ==========================================
def predict_patch(img, model, transform, crack_idx, no_idx):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    return float(p[crack_idx]), float(p[no_idx])

def predict_image(img, model, transform, crack_idx, no_idx):
    W, H = img.size
    cs = int(min(W, H) * CROP_RATIO)
    boxes = [
        (0,0,cs,cs),
        (W-cs,0,W,cs),
        (0,H-cs,cs,H),
        (W-cs,H-cs,W,H),
        ((W-cs)//2,(H-cs)//2,(W+cs)//2,(H+cs)//2)
    ]
    if USE_9_CROP:
        boxes += [
            ((W-cs)//2,0,(W+cs)//2,cs),
            ((W-cs)//2,H-cs,(W+cs)//2,H),
            (0,(H-cs)//2,cs,(H+cs)//2),
            (W-cs,(H-cs)//2,W,(H+cs)//2)
        ]
    probs = [predict_patch(img.crop(b), model, transform, crack_idx, no_idx)[0] for b in boxes]
    return max(probs), probs

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config("Stone AI Inspection", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');
body{background:#070b16;color:#e5e7eb;}
.title{font-family:Orbitron;font-size:42px;text-align:center}
.ai{color:#00e5ff}
.card{background:rgba(255,255,255,.06);padding:24px;border-radius:20px}
.badge{padding:20px 40px;border-radius:999px;font-weight:700}
.ok{color:#00e676;border:1px solid #00e676}
.bad{color:#ff4d4d;border:1px solid #ff4d4d}
.warn{color:#f59e0b;border:1px solid #f59e0b}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">STONE INSPECTION <span class="ai">AI</span></div>', unsafe_allow_html=True)

model, crack_idx, no_idx, transform = load_model()

uploaded = st.file_uploader("Upload Stone Image", type=["jpg","png","jpeg","webp","bmp","gif"])
run = st.button("START SCAN")

if run and uploaded:
    with st.spinner("AI Vision Scanning..."):
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            bar.progress(i+1)

    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext == ".gif":
        st.markdown('<div class="badge warn">❌ GIF NOT SUPPORTED</div>', unsafe_allow_html=True)
    else:
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
        with open(path,"wb") as f:
            f.write(uploaded.getvalue())

        bgr = cv2.imread(path)
        ok, lap, edge = is_stone_cv(bgr)

        if not ok:
            st.markdown('<div class="badge warn">❌ NOT A STONE</div>', unsafe_allow_html=True)
        else:
            img = Image.open(path).convert("RGB")
            crack_max, probs = predict_image(img, model, transform, crack_idx, no_idx)
            hit = sum(p >= HIT_THRESHOLD for p in probs)
            cracked = crack_max >= CRACK_THRESHOLD or hit >= HIT_K

            if cracked:
                st.markdown(
                    f'<div class="badge bad">❌ CRACK DETECTED<br>{crack_max*100:.2f}%</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="badge ok">✅ NO CRACK<br>{(1-crack_max)*100:.2f}%</div>',
                    unsafe_allow_html=True
                )

            st.image(path, use_container_width=True)

st.caption("© 2026 Stone AI Inspection System")

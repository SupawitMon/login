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

# =====================================================
# CONFIG (AI CORE - ของคุณเดิม)
# =====================================================
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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Stone AI Inspection",
    page_icon="🪨",
    layout="wide"
)

# =====================================================
# PREMIUM UI CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(-45deg, #0b1423, #0f1b2e, #0b1423);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
    padding: 32px;
    border-radius: 20px;
    box-shadow: 0 0 40px rgba(0,255,255,0.08);
}

.title {
    font-family: 'Orbitron', sans-serif;
    font-size: 48px;
    text-align:center;
    background: linear-gradient(270deg,#00bfff,#00ffcc,#8b5cf6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle {
    text-align:center;
    opacity:0.75;
    margin-bottom:40px;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    padding:22px;
    border-radius:16px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
}

.metric-label { font-size:14px; opacity:0.6; }
.metric-value { font-size:28px; font-weight:600; }

.progress-bar {
    height:14px;
    border-radius:20px;
    background:rgba(255,255,255,0.12);
    overflow:hidden;
}

.progress-fill {
    height:100%;
    transition:width 1.2s ease;
}

.success { background:linear-gradient(90deg,#00e676,#00ffcc); }
.danger  { background:linear-gradient(90deg,#ff5252,#ff1744); }
.warning { background:linear-gradient(90deg,#facc15,#f59e0b); }

.footer {
    text-align:center;
    margin-top:50px;
    opacity:0.4;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="title">Stone Defect Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Vision Inspection System</div>', unsafe_allow_html=True)

# =====================================================
# MODEL LOADER (เดิม)
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for c in r.iter_content(1024*1024):
                f.write(c)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    class_to_idx = ckpt["class_to_idx"]
    crack_idx = class_to_idx["Crack"]
    no_crack_idx = class_to_idx["No Crack"]

    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return model, crack_idx, no_crack_idx, transform

model, crack_idx, no_crack_idx, transform = load_model()

# =====================================================
# AI HELPERS (สั้นลงแต่ logic เดิม)
# =====================================================
def predict(pil):
    x = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(x),1)[0]
    return float(p[crack_idx]), float(p[no_crack_idx])

# =====================================================
# MAIN UI
# =====================================================
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Stone Image", type=["jpg","png","jpeg","webp","bmp"])
    run = st.button("🔍 Scan Stone", use_container_width=True)

    if run and uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("AI Processing..."):
            time.sleep(0.5)
            crack_p, no_p = predict(img)
            crack = crack_p >= CRACK_THRESHOLD
            confidence = round((crack_p if crack else no_p)*100,2)

        bar_class = "danger" if crack else "success"
        label = "❌ Crack Detected" if crack else "✅ No Crack Detected"

        st.markdown(f"### {label} ({confidence}%)")

        st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-fill {bar_class}" style="width:{confidence}%"></div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Crack</div><div class="metric-value">{crack}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">AI Confidence</div><div class="metric-value">{confidence}%</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value">EffNet-B3</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">© 2026 Stone AI Inspection</div>', unsafe_allow_html=True)

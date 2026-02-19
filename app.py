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
# CONFIG (คงเดิม ห้ามยุ่ง)
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
# UI CONFIG
# ==========================================
st.set_page_config(
    page_title="Stone AI Inspection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# 🔥 PRO UI STYLE (อย่างเดียว ไม่ยุ่งระบบ)
# ===============================
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}

html, body {
    background:#0b1423;
    color:white;
    font-family:'Inter', sans-serif;
}

/* Grid background */
body::before{
    content:"";
    position:fixed;
    inset:0;
    background-image:
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size:40px 40px;
    z-index:-1;
}

/* Main glass container */
.main > div{
    max-width:1100px;
    margin:auto;
    padding:42px;
    background:rgba(255,255,255,0.05);
    border-radius:26px;
    backdrop-filter:blur(22px);
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.04),
        0 40px 120px rgba(0,0,0,0.6);
}

/* Title */
.bigTitle{
    text-align:center;
    font-size:42px;
    font-weight:800;
    letter-spacing:1px;
}
.bigTitle span{
    background:linear-gradient(270deg,#00bfff,#00ffcc,#8b5cf6);
    background-size:600% 600%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:flow 6s ease infinite;
}
@keyframes flow{
    0%{background-position:0%}
    50%{background-position:100%}
    100%{background-position:0%}
}

.subTitle{
    text-align:center;
    opacity:0.75;
    margin:14px 0 36px 0;
}

/* Buttons */
.stButton>button{
    background:linear-gradient(90deg,#00bfff,#00ffcc);
    color:black;
    border:none;
    padding:14px 22px;
    border-radius:14px;
    font-weight:700;
    transition:0.3s;
}
.stButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 0 25px rgba(0,255,255,0.45);
}

/* Badge */
.badge{
    margin:24px auto;
    padding:18px 36px;
    border-radius:999px;
    font-size:20px;
    font-weight:700;
    display:inline-block;
    animation:pop .4s ease;
}
@keyframes pop{
    from{transform:scale(0.9);opacity:0}
    to{transform:scale(1);opacity:1}
}
.success{color:#16e68a;border:1px solid #16e68a;}
.danger{color:#ff5252;border:1px solid #ff5252;animation:shake .35s;}
.warning{color:#ff9800;border:1px solid #ff9800;}

@keyframes shake{
    0%{transform:translateX(0)}
    25%{transform:translateX(-6px)}
    50%{transform:translateX(6px)}
    75%{transform:translateX(-6px)}
    100%{transform:translateX(0)}
}

/* Progress */
.stProgress > div > div{
    background:linear-gradient(90deg,#00ffcc,#00bfff) !important;
}

/* Images */
img{
    border-radius:18px;
    box-shadow:0 30px 60px rgba(0,0,0,0.6);
}

/* Metrics */
[data-testid="stMetric"]{
    background:rgba(255,255,255,0.04);
    padding:18px;
    border-radius:18px;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,0.04);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# (ตั้งแต่ตรงนี้ลงไป = โค้ดคุณเดิม 100%)
# ==========================================

# --------- โหลดโมเดล ----------
@st.cache_resource(show_spinner=False)
def load_model_and_meta():
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

model, crack_idx, no_crack_idx, transform = load_model_and_meta()

# ==========================================
# UI
# ==========================================
st.markdown('<div class="bigTitle">Stone Defect Detection <span>AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subTitle">ระบบตรวจสอบคุณภาพของหินด้วย AI Vision</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("อัปโหลดรูปหิน", type=["jpg","png","jpeg","webp","bmp","gif"])
run_btn = st.button("ตรวจสอบคุณภาพ", use_container_width=True)

if uploaded and run_btn:
    img = Image.open(uploaded).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = torch.softmax(model(x), dim=1)[0]

    crack_p = float(out[crack_idx])
    no_p = float(out[no_crack_idx])

    crack = crack_p >= CRACK_THRESHOLD
    confidence = (crack_p if crack else no_p) * 100

    badge = "danger" if crack else "success"
    text = "❌ พบรอยแตก" if crack else "✅ ไม่พบรอยแตก"

    st.markdown(
        f'<div class="badge {badge}">{text}<br>ความมั่นใจ {confidence:.2f}%</div>',
        unsafe_allow_html=True
    )

    st.progress(confidence/100)

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("ภาพต้นฉบับ")
        st.image(img, use_container_width=True)
    with c2:
        st.subheader("ผลการตรวจจับ")
        st.image(img, use_container_width=True)

    m1,m2,m3 = st.columns(3)
    m1.metric("Crack Count", int(crack))
    m2.metric("Processing Time", "—")
    m3.metric("AI Confidence", f"{confidence:.2f}%")

st.caption("© 2026 Stone AI Inspection | Advanced Vision Technology")

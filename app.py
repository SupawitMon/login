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

# ================= CONFIG =================
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

# ================= MODEL =================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    r = requests.get(HF_MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)

@st.cache_resource
def load_model():
    ensure_model()
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    class_to_idx = ckpt["class_to_idx"]
    crack_idx = class_to_idx["Crack"]
    no_crack_idx = class_to_idx["No Crack"]

    IMG_SIZE = int(ckpt.get("img_size", 300))

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return model, transform, crack_idx, no_crack_idx

# ================= AI =================
def predict(pil_img, model, transform, crack_idx, no_crack_idx):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return float(probs[crack_idx]), float(probs[no_crack_idx])

# ================= UI =================
st.set_page_config(page_title="Stone AI Inspection", layout="wide")

# ---- Inject CSS ----
st.markdown("""
<style>
:root{
 --bg:#0b1423;
 --card:rgba(255,255,255,0.05);
 --text:white;
 --accent1:#00bfff;
 --accent2:#00ffcc;
 --success:#00e676;
 --danger:#ff5252;
 --warning:#ff9800;
}

html, body, [data-testid="stAppViewContainer"]{
 background:var(--bg)!important;
 color:var(--text)!important;
 font-family:'Inter',sans-serif!important;
}

.block-container{max-width:1100px;}

.containerCard{
 margin:30px auto;
 padding:40px;
 border-radius:25px;
 background:var(--card);
 backdrop-filter:blur(25px);
}

.title{
 text-align:center;
 font-size:42px;
 font-weight:800;
}

.ai{
 background:linear-gradient(270deg,var(--accent1),var(--accent2));
 -webkit-background-clip:text;
 -webkit-text-fill-color:transparent;
}

.subtitle{
 text-align:center;
 opacity:.8;
 margin-bottom:25px;
}

.stButton>button{
 background:linear-gradient(90deg,var(--accent1),var(--accent2))!important;
 color:white!important;
 border-radius:10px!important;
}

.badge{
 padding:20px 35px;
 border-radius:50px;
 display:inline-block;
 margin-top:20px;
}

.success{border:1px solid var(--success);color:var(--success);}
.danger{border:1px solid var(--danger);color:var(--danger);}
.warning{border:1px solid var(--warning);color:var(--warning);}

.image-grid{
 display:flex;
 justify-content:center;
 gap:40px;
 margin-top:30px;
 flex-wrap:wrap;
}

.image-box{
 background:var(--card);
 padding:25px;
 border-radius:20px;
}

.ai-panel{
 display:flex;
 justify-content:center;
 gap:25px;
 margin-top:30px;
 flex-wrap:wrap;
}

.ai-card{
 background:var(--card);
 padding:20px;
 border-radius:15px;
 text-align:center;
}

.footer{
 text-align:center;
 opacity:.6;
 margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ---- Layout ----
st.markdown('<div class="containerCard">', unsafe_allow_html=True)
st.markdown('<div class="title">Stone Defect Detection <span class="ai">AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ระบบตรวจสอบคุณภาพของหินด้วยเทคโนโลยี AI Vision</div>', unsafe_allow_html=True)

model, transform, crack_idx, no_crack_idx = load_model()

uploaded = st.file_uploader("อัปโหลดรูปหิน", type=["jpg","jpeg","png","webp","bmp"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")

    crack_p, no_p = predict(pil_img, model, transform, crack_idx, no_crack_idx)
    crack = crack_p > CRACK_THRESHOLD
    confidence = round((crack_p if crack else no_p)*100,2)

    badge_class = "danger" if crack else "success"
    result_text = "❌ พบรอยแตก" if crack else "✅ ไม่พบรอยแตก"

    st.markdown(f'<div style="text-align:center;"><div class="badge {badge_class}">{result_text}<br>ความมั่นใจ {confidence}%</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="image-grid">', unsafe_allow_html=True)
    st.markdown('<div class="image-box">', unsafe_allow_html=True)
    st.image(pil_img, width=340)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-card"><div>AI Confidence</div><div style="font-size:22px;">{confidence}%</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">© 2026 Stone AI Inspection</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Stone AI Inspection",
    page_icon="🪨",
    layout="wide"
)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import time
from huggingface_hub import hf_hub_download

# ==========================
# AI SETTINGS (โหดจริง)
# ==========================
CRACK_THRESHOLD = 0.55     # crack แรงพอ = ติดทันที
HIT_THRESHOLD   = 0.42     # crack เบาแต่หลายจุด
HIT_K           = 2        # ต้องเจอ ≥2 crop

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Mon2948/best_model",
        filename="best_model.pth"
    )

    ckpt = torch.load(model_path, map_location="cpu")

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    crack_idx = class_to_idx["Crack"]

    return model, crack_idx

model, CRACK_IDX = load_model()

# ==========================
# TRANSFORM
# ==========================
base_tf = transforms.Compose([
    transforms.Resize((340, 340)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================
# MULTI CROP
# ==========================
def multi_crop(img, size=300):
    w, h = img.size
    crops = [
        img.crop((0, 0, size, size)),
        img.crop((w-size, 0, w, size)),
        img.crop((0, h-size, size, h)),
        img.crop((w-size, h-size, w, h)),
        img.crop(((w-size)//2, (h-size)//2,
                  (w+size)//2, (h+size)//2))
    ]
    return crops

# ==========================
# AI PREDICT (สมองหลัก)
# ==========================
def predict(image: Image.Image):
    crops = multi_crop(image)
    crack_scores = []

    with torch.no_grad():
        for crop in crops:
            x = base_tf(crop).unsqueeze(0)
            out = model(x)
            prob = torch.softmax(out, dim=1)[0][CRACK_IDX].item()
            crack_scores.append(prob)

    crack_max = max(crack_scores)
    hit_count = sum(p >= HIT_THRESHOLD for p in crack_scores)

    is_crack = (
        crack_max >= CRACK_THRESHOLD
        or hit_count >= HIT_K
    )

    return is_crack, crack_max, crack_scores

# ==========================
# UI
# ==========================
st.markdown("<h1 style='text-align:center'>Stone Defect Detection AI</h1>",
            unsafe_allow_html=True)
st.markdown("<center>Industrial Vision Inspection</center><br>",
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload Stone Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_column_width=True)

    with col2:
        if st.button("🔥 START AI SCAN"):
            bar = st.progress(0)
            for i in range(100):
                time.sleep(0.003)
                bar.progress(i+1)

            is_crack, crack_max, scores = predict(img)

            conf = round(crack_max * 100, 2)
            color = "#ff3b3b" if is_crack else "#00ffcc"

            st.markdown(
                f"""
                <div style="
                width:180px;height:180px;
                border-radius:50%;
                margin:auto;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:26px;
                font-weight:bold;
                background:
                  radial-gradient(circle,#0b1423 45%, transparent 46%),
                  conic-gradient({color} {int(conf)}%, #1e293b {int(conf)}%);
                box-shadow:0 0 40px {color};
                ">
                {conf}%
                </div>
                """,
                unsafe_allow_html=True
            )

            if is_crack:
                st.error(f"🚨 CRACK DETECTED ({conf}%)")
            else:
                st.success(f"✅ GOOD STONE ({conf}%)")

            st.caption(f"crop scores: {['%.2f'%s for s in scores]}")


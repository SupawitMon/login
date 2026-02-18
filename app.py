import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 300
CRACK_THRESHOLD = 0.6

st.set_page_config(page_title="Stone Crack AI", layout="centered")
st.title("🪨 Stone Crack Inspection AI")

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    ckpt = torch.load("best_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt["class_to_idx"]

model, class_to_idx = load_model()
crack_idx = class_to_idx["Crack"]
no_idx = class_to_idx["No Crack"]

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =====================
# AI PREDICT
# =====================
def predict(img: Image.Image):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return float(probs[crack_idx]), float(probs[no_idx])

# =====================
# UI
# =====================
file = st.file_uploader(
    "อัปโหลดรูปหิน",
    type=["jpg","jpeg","png","webp"]
)

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="ภาพต้นฉบับ", use_column_width=True)

    with st.spinner("กำลังวิเคราะห์..."):
        crack_p, no_p = predict(img)

    if crack_p >= CRACK_THRESHOLD:
        st.error(f"❌ พบรอยแตก ({crack_p*100:.2f}%)")
    else:
        st.success(f"✅ ไม่พบรอยแตก ({no_p*100:.2f}%)")

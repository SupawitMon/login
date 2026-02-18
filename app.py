import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

CLASS_NAMES = ["Crack", "No Crack"]
CRACK_IDX = 0
NO_IDX = 1

# ===============================
# DOWNLOAD MODEL (SAFE)
# ===============================
def download_model():
    if os.path.exists(MODEL_PATH):
        return

    with st.spinner("📥 Downloading AI model..."):
        r = requests.get(MODEL_URL, timeout=30)
        r.raise_for_status()

        # กันโหลด HTML page
        if b"<html" in r.content[:200].lower():
            raise RuntimeError("❌ Downloaded file is HTML, not a PyTorch model")

        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    download_model()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

# ===============================
# IMAGE PREPROCESS
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def preprocess_image(img_pil):
    return transform(img_pil).unsqueeze(0).to(DEVICE)

# ===============================
# PREDICT
# ===============================
def predict(img_pil, model):
    x = preprocess_image(img_pil)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    crack_p = probs[CRACK_IDX].item()
    no_p = probs[NO_IDX].item()

    if crack_p > no_p:
        return "❌ หินแตก", crack_p
    else:
        return "✅ หินดี", no_p

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(
    page_title="Stone Crack AI",
    layout="centered"
)

st.title("🪨 Stone Crack AI Inspection")
st.caption("ResNet18 • PyTorch • Streamlit")

model = load_model()

uploaded = st.file_uploader(
    "📷 อัปโหลดภาพหิน",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="ภาพต้นฉบับ", use_column_width=True)

    label, confidence = predict(img, model)

    st.markdown("---")
    st.subheader("📊 ผลการทำนาย")
    st.metric(
        label=label,
        value=f"{confidence*100:.2f} %"
    )

    if "แตก" in label:
        st.error("พบรอยแตกในหิน")
    else:
        st.success("ไม่พบรอยแตก")

st.markdown("---")
st.caption("© Stone Crack AI")

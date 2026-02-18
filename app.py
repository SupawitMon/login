# ================================
# Stone Crack Inspection - Streamlit
# ================================

import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image

import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as F

# ================================
# CONFIG
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model.pth"

IMG_SIZE = 300
MIN_RESIZE = 320

CRACK_THRESHOLD = 0.58
HIT_THRESHOLD = 0.48
HIT_K = 2

USE_FIVE_CROP = True

STONE_LAP_MIN = 90.0
STONE_EDGE_MIN = 0.015

st.set_page_config(page_title="Stone AI Inspection", layout="centered")

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    class_to_idx = ckpt["class_to_idx"]
    crack_idx = class_to_idx["Crack"]
    no_idx = class_to_idx["No Crack"]

    return model, crack_idx, no_idx


model, crack_idx, no_crack_idx = load_model()

# ================================
# TRANSFORMS
# ================================
base_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

five_crop = transforms.FiveCrop(IMG_SIZE)

# ================================
# CV STONE GATE
# ================================
def is_stone_cv(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return (lap >= STONE_LAP_MIN and edge_density >= STONE_EDGE_MIN), lap, edge_density

# ================================
# SAFE FIVE CROP (🔥 ตัวแก้พัง)
# ================================
def safe_five_crop(pil_img: Image.Image):
    w, h = pil_img.size

    if min(w, h) < IMG_SIZE:
        scale = IMG_SIZE / min(w, h)
        pil_img = F.resize(
            pil_img,
            (int(h * scale) + 1, int(w * scale) + 1)
        )

    pil_img = F.resize(pil_img, MIN_RESIZE)
    crops = five_crop(pil_img)

    tensors = [base_tf(c) for c in crops]
    return torch.stack(tensors)

# ================================
# PREDICT
# ================================
@torch.no_grad()
def predict(pil_img):
    if USE_FIVE_CROP:
        batch = safe_five_crop(pil_img).to(DEVICE)
        logits = model(batch)
        probs = torch.softmax(logits, 1)

        crack_probs = probs[:, crack_idx].cpu().numpy().tolist()
        crack_max = max(crack_probs)
        hits = sum(p >= HIT_THRESHOLD for p in crack_probs)

        is_crack = crack_max >= CRACK_THRESHOLD or hits >= HIT_K
        conf = crack_max if is_crack else max(probs[:, no_crack_idx]).item()

        return is_crack, conf * 100, crack_probs

    else:
        x = base_tf(pil_img).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(x), 1)[0]
        is_crack = probs[crack_idx] >= CRACK_THRESHOLD
        conf = probs[crack_idx] if is_crack else probs[no_crack_idx]
        return bool(is_crack), conf.item() * 100, [probs[crack_idx].item()]

# ================================
# UI
# ================================
st.title("🪨 Stone Crack Inspection AI")
st.caption("EfficientNet-B3 | FiveCrop | Industrial Logic")

uploaded = st.file_uploader(
    "อัปโหลดภาพหิน (jpg / png / webp)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Original Image", use_column_width=True)

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    ok_stone, lap, edge = is_stone_cv(bgr)

    if not ok_stone:
        st.error("❌ ไม่ใช่หิน")
        st.write(f"Laplacian: {lap:.1f}")
        st.write(f"Edge density: {edge:.4f}")
        st.stop()

    with st.spinner("AI กำลังตรวจสอบ..."):
        is_crack, conf, crops = predict(pil_img)

    if is_crack:
        st.error(f"❌ พบรอยแตก ({conf:.2f}%)")
    else:
        st.success(f"✅ ไม่พบรอยแตก ({conf:.2f}%)")

    with st.expander("🔍 Crack probabilities (per crop)"):
        for i, p in enumerate(crops):
            st.write(f"Crop {i+1}: {p:.3f}")

st.markdown("---")
st.caption("ระบบนี้ใช้ logic แบบโรงงาน (ไม่โชว์มั่ว ไม่เดา)")

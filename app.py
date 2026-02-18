import os
import requests
import streamlit as st
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ==============================
# CONFIG
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

IMG_SIZE = 300

# 🔒 โหมด conservative (ลดมั่ว)
CRACK_THRESHOLD = 0.65   # ต้องมั่นใจจริงถึงจะฟันว่าแตก
HIT_THRESHOLD = 0.55
HIT_K = 2

# ==============================
# DOWNLOAD MODEL
# ==============================
def download_model():
    if os.path.exists(MODEL_PATH):
        return
    with st.spinner("📥 Downloading AI model..."):
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    download_model()

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    state = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    class_to_idx = ckpt["class_to_idx"]
    crack_idx = class_to_idx["Crack"]
    no_idx = class_to_idx["No Crack"]

    return model, crack_idx, no_idx

model, CRACK_IDX, NO_IDX = load_model()

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# CV STONE GATE (กันภาพมั่ว)
# ==============================
def is_stone(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    return lap > 80 and edge_density > 0.01

# ==============================
# SAFE MULTI-CROP (ไม่พังรูปเล็ก)
# ==============================
def multi_crop_pil(img):
    W, H = img.size
    s = int(min(W, H) * 0.75)
    s = max(64, s)

    crops = []
    xs = [0, W - s, (W - s) // 2]
    ys = [0, H - s, (H - s) // 2]

    for x in xs:
        for y in ys:
            if x >= 0 and y >= 0 and x + s <= W and y + s <= H:
                crops.append(img.crop((x, y, x + s, y + s)))

    return crops if crops else [img]

# ==============================
# PREDICT
# ==============================
def predict(img_pil):
    crops = multi_crop_pil(img_pil)

    crack_probs = []
    with torch.no_grad():
        for c in crops:
            x = transform(c).unsqueeze(0).to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            crack_probs.append(float(probs[CRACK_IDX]))

    crack_max = max(crack_probs)
    hit_count = sum(p >= HIT_THRESHOLD for p in crack_probs)

    is_crack = (crack_max >= CRACK_THRESHOLD) and (hit_count >= HIT_K)

    return is_crack, crack_max, hit_count

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Stone Crack Inspection AI", layout="centered")
st.title("🪨 Stone Crack Inspection AI")

uploaded = st.file_uploader(
    "อัปโหลดภาพหิน (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="ภาพที่อัปโหลด", use_container_width=True)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if not is_stone(bgr):
        st.error("❌ ภาพนี้ไม่ใช่หิน (CV gate)")
    else:
        with st.spinner("🔍 AI กำลังวิเคราะห์..."):
            is_crack, crack_prob, hits = predict(img)

        conf = round(crack_prob * 100, 2)

        if is_crack:
            st.error(f"❌ พบรอยแตก\n\nConfidence: {conf}%\nHits: {hits}")
        else:
            st.success(f"✅ ไม่พบรอยแตก\n\nConfidence: {100-conf}%")

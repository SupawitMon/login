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
import gdown
import torch.serialization

# ==========================================
# CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "15dY4OBZ_pii_NR8FnRpESjIpZ8omsXtH"

CRACK_THRESHOLD = 0.58
HIT_THRESHOLD   = 0.48
HIT_K           = 2

USE_MULTI_CROP = True
CROP_RATIO = 0.75
USE_9_CROP = True

STONE_LAP_MIN  = 90.0
STONE_EDGE_MIN = 0.015

ALLOWED_EXT = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# ==========================================
# Download model if missing
# ==========================================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

ensure_model()

# ==========================================
# LOAD MODEL
# ==========================================
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

model = models.efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()

class_to_idx = ckpt["class_to_idx"]
crack_idx = class_to_idx["Crack"]
no_crack_idx = class_to_idx["No Crack"]

IMG_SIZE = int(ckpt.get("img_size", 300))

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# CV Stone Gate
# ==========================================
def is_stone_cv(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / (bgr_img.shape[0] * bgr_img.shape[1]))
    return (lap_var >= STONE_LAP_MIN) and (edge_density >= STONE_EDGE_MIN)

# ==========================================
# AI Predict
# ==========================================
def _predict_probs(pil_img):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return float(probs[crack_idx]), float(probs[no_crack_idx])

def predict_image_ai(pil_img):
    if not USE_MULTI_CROP:
        return _predict_probs(pil_img)

    W, H = pil_img.size
    crop_size = int(min(W, H) * CROP_RATIO)
    crop_size = max(32, crop_size)

    boxes = [
        (0,0,crop_size,crop_size),
        (W-crop_size,0,W,crop_size),
        (0,H-crop_size,crop_size,H),
        (W-crop_size,H-crop_size,W,H),
        ((W-crop_size)//2,(H-crop_size)//2,
         (W+crop_size)//2,(H+crop_size)//2)
    ]

    crack_probs = []
    no_probs = []

    for b in boxes:
        patch = pil_img.crop(b)
        c,n = _predict_probs(patch)
        crack_probs.append(c)
        no_probs.append(n)

    return max(crack_probs), max(no_probs)

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("🪨 Stone Crack Inspection AI")

uploaded_file = st.file_uploader(
    "อัปโหลดรูปหิน",
    type=["jpg","jpeg","png","webp","bmp"]
)

if uploaded_file is not None:

    start_time = time.time()

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if not is_stone_cv(bgr):
        st.error("❌ ไม่ใช่หิน")
    else:
        crack_prob, no_crack_prob = predict_image_ai(image)

        crack_hits = crack_prob >= CRACK_THRESHOLD
        is_crack = crack_hits

        confidence = round((crack_prob if is_crack else no_crack_prob)*100,2)

        if is_crack:
            st.error(f"❌ พบรอยแตก ({confidence}%)")
        else:
            st.success(f"✅ ไม่พบรอยแตก ({confidence}%)")

    st.write("Processing time:", round(time.time()-start_time,3), "sec")

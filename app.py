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
import time
from huggingface_hub import hf_hub_download

# ==========================
# DOWNLOAD MODEL
# ==========================
@st.cache_resource
def get_model_path():
    return hf_hub_download(
        repo_id="Mon2948/best_model",
        filename="best_model.pth"
    )

MODEL_PATH = get_model_path()

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class


model, idx_to_class = load_model()

# ==========================
# TRANSFORM (ต้องตรงกับตอน TRAIN)
# ==========================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ===============================
# UI
# ===============================
st.title("Stone Defect Detection AI")
st.write("Industrial Vision Inspection System")

uploaded = st.file_uploader("Upload Stone Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, use_column_width=True)

    with col2:
        if st.button("Start AI Scan"):

            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.003)
                progress.progress(i+1)

            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item()

            pred_class = idx_to_class[pred_idx]
            percent = int(confidence * 100)

            is_crack = pred_class.lower() == "crack"

            ring_color = "#ff3b3b" if is_crack else "#00ffcc"

            ring_html = f"""
            <div style="
                width:180px;
                height:180px;
                border-radius:50%;
                margin:auto;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:24px;
                font-weight:bold;
                background:
                radial-gradient(circle,#0f172a 45%, transparent 46%),
                conic-gradient({ring_color} {percent}%, #1e293b {percent}%);
                box-shadow:0 0 40px {ring_color};">
                {percent}%
            </div>
            """

            st.markdown(ring_html, unsafe_allow_html=True)

            if is_crack:
                st.error(f"🚨 Crack Detected ({percent}%)")
            else:
                st.success(f"✅ No Crack Detected ({percent}%)")

st.markdown("---")
st.caption("© 2026 Stone AI Inspection")

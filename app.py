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
# LOAD MODEL + CLASS MAP
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
# TRANSFORM
# ==========================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
html, body {
    background: radial-gradient(circle at 20% 20%, #0f2027, #0b1423 60%);
    color: white;
}
.title {
    font-size: 48px;
    text-align: center;
    font-weight: 700;
    background: linear-gradient(90deg,#00f5ff,#00ffcc,#8b5cf6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.glass {
    background: rgba(255,255,255,0.05);
    padding: 40px;
    border-radius: 25px;
}
.result-ring {
    width:180px;
    height:180px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:24px;
    font-weight:bold;
    margin:auto;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown('<div class="title">Stone Defect Detection AI</div>', unsafe_allow_html=True)
st.markdown("<center>Industrial Vision Inspection System</center><br>", unsafe_allow_html=True)

# ===============================
# MAIN
# ===============================
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Stone Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            if st.button("Start AI Scan"):

                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.003)
                    bar.progress(i + 1)

                img = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(img)
                    prob = torch.softmax(output, dim=1)[0]

                # ===== CLASS PROB =====
                crack_idx = [k for k,v in idx_to_class.items() if v.lower() == "crack"][0]
                good_idx  = [k for k,v in idx_to_class.items() if v.lower() != "crack"][0]

                crack_prob = prob[crack_idx].item()
                good_prob  = prob[good_idx].item()

                # ===== INDUSTRIAL DECISION LOGIC =====
                # โหด แต่ไม่มั่ว
                if crack_prob >= 0.70:
                    is_crack = True
                elif crack_prob >= 0.60 and crack_prob > good_prob + 0.15:
                    is_crack = True
                else:
                    is_crack = False

                confidence = crack_prob if is_crack else good_prob
                confidence = round(confidence * 100, 2)
                percent = int(confidence)

                ring_color = "#ff3b3b" if is_crack else "#00ffcc"

                ring_html = f"""
                <div class="result-ring" style="
                background:
                radial-gradient(circle,#0b1423 45%, transparent 46%),
                conic-gradient({ring_color} {percent}%, #1e293b {percent}%);
                box-shadow:0 0 40px {ring_color};">
                {confidence}%
                </div>
                """

                st.markdown(ring_html, unsafe_allow_html=True)

                # DEBUG (เปิดดูได้)
                st.caption(
                    f"Crack={crack_prob:.3f} | Good={good_prob:.3f}"
                )

                if is_crack:
                    st.error(f"🚨 Crack Detected ({confidence}%)")
                else:
                    st.success(f"✅ No Crack Detected ({confidence}%)")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<center style="opacity:0.4;margin-top:60px;">© 2026 Stone AI Inspection</center>',
    unsafe_allow_html=True
)

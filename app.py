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
import os
import requests

# ==========================
# MODEL DOWNLOAD CONFIG
# ==========================
MODEL_PATH = "best_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=15dY4OBZ_pii_NR8FnRpESjIpZ8omsXtH"

def download_model():
    with st.spinner("Downloading AI Model (first run only)..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

if not os.path.exists(MODEL_PATH):
    download_model()

# ==========================
# LOCKED SETTINGS
# ==========================
CRACK_THRESHOLD = 0.58
HIT_THRESHOLD   = 0.48
HIT_K           = 2



# ==========================
# LOAD MODEL (ถูกต้องตาม training script)
# ==========================
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

model = load_model()

# ==========================
# TRANSFORM (ต้อง normalize เหมือนตอน train)
# ==========================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ===============================
# ULTRA PREMIUM CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at 20% 20%, #0f2027, #0b1423 60%);
    color: white;
}

.title {
    font-family: 'Orbitron', sans-serif;
    font-size: 50px;
    text-align:center;
    background: linear-gradient(270deg,#00f5ff,#00ffcc,#8b5cf6,#00f5ff);
    background-size:600% 600%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: flow 6s ease infinite;
}

@keyframes flow {
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    padding: 40px;
    border-radius: 25px;
    border:1px solid rgba(0,255,255,0.1);
    box-shadow: 0 0 60px rgba(0,255,255,0.08);
}

.result-ring {
    width:180px;
    height:180px;
    border-radius:50%;
    margin:auto;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:24px;
    font-weight:bold;
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
                    time.sleep(0.004)
                    progress.progress(i+1)

                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.softmax(output, dim=1)
                    crack_prob = prob[0][1].item()

                crack_max = crack_prob
                hit_count = 1 if crack_prob >= HIT_THRESHOLD else 0

                if crack_max >= CRACK_THRESHOLD:
                    final_result = True
                elif hit_count >= HIT_K:
                    final_result = True
                else:
                    final_result = False

                confidence = round(crack_max * 100, 2)
                percent = int(confidence)

                ring_color = "#ff3b3b" if final_result else "#00ffcc"

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

                if final_result:
                    st.error(f"🚨 Crack Detected ({confidence}%)")
                else:
                    st.success(f"✅ No Crack Detected ({confidence}%)")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<center style="opacity:0.4;margin-top:60px;">© 2026 Stone AI Inspection</center>', unsafe_allow_html=True)


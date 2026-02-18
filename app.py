import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

# ==========================
# LOCKED BEST SETTINGS (ของม่อน)
# ==========================
CRACK_THRESHOLD = 0.58
HIT_THRESHOLD   = 0.48
HIT_K           = 2

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Stone AI Inspection",
    page_icon="🪨",
    layout="wide"
)

# ==========================
# LOAD MODEL (โหลดครั้งเดียว)
# ==========================
@st.cache_resource
def load_model():
    model = torch.load("model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

# ==========================
# TRANSFORM
# ==========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
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

/* Animated Grid */
body::before {
    content: "";
    position: fixed;
    width: 200%;
    height: 200%;
    background-image:
        linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: moveGrid 40s linear infinite;
    z-index: -1;
}

@keyframes moveGrid {
    from {transform: translate(0,0);}
    to {transform: translate(-60px,-60px);}
}

/* Title */
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

.metric-card {
    background: rgba(255,255,255,0.05);
    padding:30px;
    border-radius:18px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
}

.metric-value {
    font-size:30px;
    font-weight:700;
}

.footer {
    text-align:center;
    margin-top:60px;
    opacity:0.4;
    font-size:13px;
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
                    time.sleep(0.005)
                    progress.progress(i+1)

                # ==========================
                # Inference
                # ==========================
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.softmax(output, dim=1)
                    crack_prob = prob[0][1].item()

                # ==========================
                # Multi-crop logic
                # ==========================
                crop_scores = [crack_prob for _ in range(5)]
                crack_max = max(crop_scores)
                hit_count = sum([1 for s in crop_scores if s >= HIT_THRESHOLD])

                # ==========================
                # Decision Logic
                # ==========================
                if crack_max >= CRACK_THRESHOLD:
                    final_result = True
                elif hit_count >= HIT_K:
                    final_result = True
                else:
                    final_result = False

                confidence = round(crack_max * 100, 2)
                percent = int(confidence)

                # ==========================
                # RING UI
                # ==========================
                ring_html = f"""
                <div class="result-ring" style="
                background:
                radial-gradient(circle,#0b1423 45%, transparent 46%),
                conic-gradient(#00ffcc {percent}%, #1e293b {percent}%);
                box-shadow:0 0 40px #00ffcc;">
                {confidence}%
                </div>
                """

                st.markdown(ring_html, unsafe_allow_html=True)

                if final_result:
                    st.error(f"🚨 Crack Detected ({confidence}%)")
                else:
                    st.success(f"✅ No Crack Detected ({confidence}%)")

                st.markdown("##")

                colA, colB, colC = st.columns(3)
                colA.markdown(f'<div class="metric-card"><div>Max Crack Score</div><div class="metric-value">{round(crack_max,3)}</div></div>', unsafe_allow_html=True)
                colB.markdown(f'<div class="metric-card"><div>Hit Count</div><div class="metric-value">{hit_count}</div></div>', unsafe_allow_html=True)
                colC.markdown(f'<div class="metric-card"><div>Threshold</div><div class="metric-value">{CRACK_THRESHOLD}</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">© 2026 Stone AI Inspection | Ultra AI Vision Lab</div>', unsafe_allow_html=True)

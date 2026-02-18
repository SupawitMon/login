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
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class_to_idx = ckpt["class_to_idx"]
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
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# ===============================
# UI
# ===============================
st.markdown("""
<style>
body { background:#0b1423; color:white; }
.title {
    font-size:46px;
    text-align:center;
    font-weight:700;
    background:linear-gradient(90deg,#00f5ff,#8b5cf6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.glass {
    background:rgba(255,255,255,0.05);
    padding:40px;
    border-radius:25px;
}
.result-ring {
    width:180px;height:180px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:24px;font-weight:bold;margin:auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Stone Defect Detection AI</div>', unsafe_allow_html=True)
st.markdown("<center>Industrial Vision Inspection</center><br>", unsafe_allow_html=True)

# ===============================
# MAIN
# ===============================
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    file = st.file_uploader("Upload Stone Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        c1, c2 = st.columns(2)

        with c1:
            st.image(img, use_column_width=True)

        with c2:
            if st.button("Start AI Scan"):

                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.003)
                    bar.progress(i+1)

                x = transform(img).unsqueeze(0)

                with torch.no_grad():
                    out = model(x)
                    prob = torch.softmax(out, dim=1)[0]

                # ===== CLASS INDEX =====
                crack_idx = [i for i,v in idx_to_class.items() if v.lower()=="crack"][0]
                good_idx  = [i for i,v in idx_to_class.items() if v.lower()!="crack"][0]

                crack_p = prob[crack_idx].item()
                good_p  = prob[good_idx].item()

                # ==========================
                # 🔥 FAIL-SAFE INDUSTRIAL LOGIC
                # ==========================
                if crack_p >= 0.65:
                    is_crack = True
                elif crack_p >= 0.45:
                    is_crack = True
                elif crack_p >= good_p - 0.10:
                    is_crack = True
                else:
                    is_crack = False

                conf = crack_p if is_crack else good_p
                conf = round(conf * 100, 2)
                percent = int(conf)

                color = "#ff3b3b" if is_crack else "#00ffcc"

                st.markdown(f"""
                <div class="result-ring" style="
                background:
                radial-gradient(circle,#0b1423 45%, transparent 46%),
                conic-gradient({color} {percent}%, #1e293b {percent}%);
                box-shadow:0 0 40px {color};">
                {conf}%
                </div>
                """, unsafe_allow_html=True)

                st.caption(f"DEBUG → Crack={crack_p:.3f} | Good={good_p:.3f}")

                if is_crack:
                    st.error(f"🚨 Crack Detected ({conf}%)")
                else:
                    st.success(f"✅ Good Stone ({conf}%)")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<center style='opacity:0.4;margin-top:50px;'>© 2026 Stone AI</center>", unsafe_allow_html=True)

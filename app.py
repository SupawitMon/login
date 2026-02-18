import streamlit as st
st.set_page_config(page_title="Stone AI Inspection", page_icon="🪨", layout="wide")

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import time
from huggingface_hub import hf_hub_download

# ==========================
# MODEL
# ==========================
@st.cache_resource
def get_model():
    path = hf_hub_download("Mon2948/best_model", "best_model.pth")
    ckpt = torch.load(path, map_location="cpu")

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    return model, idx_to_class

model, idx_to_class = get_model()

# ==========================
# TRANSFORMS
# ==========================
normalize = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
base = T.Compose([T.Resize((320,320)), T.ToTensor(), normalize])

five_crop = T.FiveCrop(300)

def preprocess(img):
    crops = five_crop(img)
    return torch.stack([base(c) for c in crops])

# ==========================
# UI
# ==========================
st.markdown("""
<style>
body{background:#0b1423;color:white}
.title{font-size:46px;text-align:center;font-weight:700;
background:linear-gradient(90deg,#00f5ff,#8b5cf6);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.box{background:rgba(255,255,255,.05);padding:40px;border-radius:25px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Stone Defect Detection AI</div>', unsafe_allow_html=True)
st.markdown("<center>Multi-View Industrial Inspection</center><br>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="box">', unsafe_allow_html=True)
    f = st.file_uploader("Upload Stone Image", type=["jpg","png","jpeg"])

    if f:
        img = Image.open(f).convert("RGB")
        c1, c2 = st.columns(2)

        with c1:
            st.image(img, use_column_width=True)

        with c2:
            if st.button("Start AI Scan"):

                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.002)
                    bar.progress(i+1)

                x = preprocess(img)

                with torch.no_grad():
                    out = model(x)
                    prob = torch.softmax(out, dim=1)

                crack_idx = [i for i,v in idx_to_class.items() if v.lower()=="crack"][0]
                crack_probs = prob[:, crack_idx]

                crack_hits = (crack_probs > 0.45).sum().item()
                crack_max = crack_probs.max().item()

                # ==========================
                # FINAL DECISION
                # ==========================
                is_crack = crack_hits >= 2 or crack_max >= 0.65

                confidence = crack_max if is_crack else (1 - crack_max)
                confidence = round(confidence * 100, 2)

                st.caption(f"DEBUG → crack_hits={crack_hits}/5 | max={crack_max:.3f}")

                if is_crack:
                    st.error(f"🚨 Crack Detected ({confidence}%)")
                else:
                    st.success(f"✅ Good Stone ({confidence}%)")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<center style='opacity:.4;margin-top:50px'>© 2026 Stone AI</center>", unsafe_allow_html=True)

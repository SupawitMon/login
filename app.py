import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

CLASS_NAMES = ["Crack", "No Crack"]

# ===============================
# DOWNLOAD MODEL
# ===============================
def download_model():
    if os.path.exists(MODEL_PATH):
        return

    with st.spinner("📥 Downloading model..."):
        r = requests.get(MODEL_URL, timeout=30)
        r.raise_for_status()

        if b"<html" in r.content[:200].lower():
            raise RuntimeError("Downloaded file is HTML, not a model")

        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

# ===============================
# LOAD MODEL (FIXED)
# ===============================
@st.cache_resource
def load_model():
    download_model()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # 🔥 รองรับทุก format
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Invalid checkpoint format")

    # กรณีเทรนด้วย DataParallel
    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state)
    model.to(DEVICE)
    model.eval()
    return model

# ===============================
# PREPROCESS
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def predict(img, model):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]

    idx = prob.argmax().item()
    return CLASS_NAMES[idx], prob[idx].item()

# ===============================
# UI
# ===============================
st.set_page_config(page_title="Stone Crack AI", layout="centered")
st.title("🪨 Stone Crack AI Inspection")
st.caption("ResNet18 • PyTorch • Streamlit")

model = load_model()

file = st.file_uploader("📷 Upload stone image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)

    label, conf = predict(img, model)

    st.markdown("---")
    st.metric(label, f"{conf*100:.2f}%")

    if label == "Crack":
        st.error("❌ พบรอยแตก")
    else:
        st.success("✅ หินดี")

st.markdown("---")
st.caption("© Stone Crack AI")

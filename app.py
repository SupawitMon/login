import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from safetensors.torch import load_file as safe_load

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cpu")

MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

CLASS_NAMES = ["Crack", "No Crack"]

# ===============================
# DOWNLOAD MODEL
# ===============================
def download_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    st.info("📥 Downloading model from HuggingFace...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    st.success(f"✅ Model size: {size:.2f} MB")

# ===============================
# LOAD MODEL (AUTO DETECT)
# ===============================
def load_model():
    download_model()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = None

    # 🔥 TRY 1: torch.load (state_dict)
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
        st.success("✅ Loaded with torch.load")
    except Exception:
        pass

    # 🔥 TRY 2: safetensors
    if state_dict is None:
        try:
            state_dict = safe_load(MODEL_PATH)
            st.success("✅ Loaded with safetensors")
        except Exception:
            pass

    # 🔥 TRY 3: TorchScript
    if state_dict is None:
        try:
            model = torch.jit.load(MODEL_PATH, map_location="cpu")
            model.eval()
            st.success("✅ Loaded TorchScript model")
            return model
        except Exception:
            pass

    if state_dict is None:
        raise RuntimeError("❌ Cannot load model: unknown format")

    # clean key
    clean = {}
    for k, v in state_dict.items():
        clean[k.replace("module.", "")] = v

    model.load_state_dict(clean)
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
    x = transform(img).unsqueeze(0)
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

model = load_model()

file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)

    label, conf = predict(img, model)

    st.metric("Prediction", label, f"{conf*100:.2f}%")

    if label == "Crack":
        st.error("❌ Crack detected")
    else:
        st.success("✅ No crack")

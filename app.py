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
DEVICE = torch.device("cpu")  # บังคับ cpu บน Streamlit
MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["Crack", "No Crack"]

# ===============================
# FORCE DOWNLOAD MODEL
# ===============================
def download_model_force():
    # 🔥 ลบทิ้งก่อนเสมอ
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    st.info("📥 Downloading model from HuggingFace...")
    r = requests.get(MODEL_URL, stream=True, timeout=60)
    r.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    st.write(f"✅ Model size: {size_mb:.2f} MB")

    # 🔥 กัน HTML
    if size_mb < 1:
        raise RuntimeError("❌ Downloaded file is NOT a real .pth model")

# ===============================
# LOAD MODEL (NO CACHE)
# ===============================
def load_model():
    download_model_force()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False  # สำคัญ
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Invalid checkpoint format")

    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    model.load_state_dict(clean_state)
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

model = load_model()

file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)

    label, conf = predict(img, model)
    st.metric(label, f"{conf*100:.2f}%")

    if label == "Crack":
        st.error("❌ Crack detected")
    else:
        st.success("✅ No crack")

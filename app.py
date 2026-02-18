import streamlit as st
import torch
import requests
import os
from PIL import Image
import torchvision.transforms as transforms

DEVICE = torch.device("cpu")
MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

st.set_page_config(page_title="Stone Crack AI", layout="centered")

# -------------------------
# Download model
# -------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading model from HuggingFace...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.success(f"✅ Model size: {os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB")

# -------------------------
# Load model (FULL MODEL)
# -------------------------
@st.cache_resource
def load_model():
    download_model()

    model = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.eval()
    st.success("✅ Loaded FULL model (no state_dict)")
    return model

model = load_model()

# -------------------------
# Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# -------------------------
# UI
# -------------------------
st.title("🪨 Stone Crack AI Inspection")

file = st.file_uploader("Upload stone image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Input image", use_container_width=True)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]

    crack = prob[1].item() * 100
    good  = prob[0].item() * 100

    st.metric("🟢 Good Stone", f"{good:.1f}%")
    st.metric("🔴 Cracked Stone", f"{crack:.1f}%")

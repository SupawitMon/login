import os
import time
import uuid

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import streamlit as st
import streamlit.components.v1 as components
import requests


# ==========================================
# CONFIG (คงเดิม)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model download (Hugging Face) ----
MODEL_PATH = "best_model.pth"
HF_MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth?download=true"

UPLOAD_FOLDER = "static/uploads"

# ---- LOCKED BEST SETTINGS (ของม่อน) ----
CRACK_THRESHOLD = 0.58   # ด่านหลัก: crack_max >= 0.58 -> แตก
HIT_THRESHOLD   = 0.48   # ด่านรอง: ต่อ crop
HIT_K           = 2      # ต้องเจออย่างน้อย 2 crop ถึงถือว่าแตก

# ---- Multi-crop ----
USE_MULTI_CROP = True
CROP_RATIO = 0.75
USE_9_CROP = True

# ---- Stone gate (OpenCV) ----
STONE_LAP_MIN  = 90.0     # 80-140
STONE_EDGE_MIN = 0.015    # 0.01-0.03

# ---- Allowed extensions ----
# NOTE: อนุญาต gif ให้ "อัปได้" แต่จะตอบว่า "ยังไม่รองรับ GIF" ตามที่ม่อนต้องการ
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# Download model if missing
# ==========================================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    st.info("Downloading model from Hugging Face...")
    r = requests.get(HF_MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

# ==========================================
# LOAD MODEL (checkpoint dict) - cache
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model_and_meta():
    ensure_model()

    # ให้ torch load numpy scalar ได้เหมือนของเดิม (แต่ต้องกันกรณี torch ไม่มี API นี้)
    try:
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

    # รองรับ torch ที่ไม่มี weights_only
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    class_to_idx = ckpt.get("class_to_idx") if isinstance(ckpt, dict) else None
    if class_to_idx is None:
        raise RuntimeError("best_model.pth ไม่มี class_to_idx — กรุณาเซฟโมเดลเป็น dict ที่มี class_to_idx")

    CRACK_NAME = "Crack"
    NOCRACK_NAME = "No Crack"
    if CRACK_NAME not in class_to_idx or NOCRACK_NAME not in class_to_idx:
        raise RuntimeError(f"class_to_idx ไม่เจอ '{CRACK_NAME}' หรือ '{NOCRACK_NAME}' -> {class_to_idx}")

    crack_idx = class_to_idx[CRACK_NAME]
    no_crack_idx = class_to_idx[NOCRACK_NAME]

    IMG_SIZE = int(ckpt.get("img_size", 300)) if isinstance(ckpt, dict) else 300

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, class_to_idx, crack_idx, no_crack_idx, IMG_SIZE, transform


# ===============================
# 🔍 CV Stone Gate (คงเดิม)
# ===============================
def is_stone_cv(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / (bgr_img.shape[0] * bgr_img.shape[1]))

    is_stone = (lap_var >= STONE_LAP_MIN) and (edge_density >= STONE_EDGE_MIN)
    return is_stone, float(lap_var), float(edge_density)


def stone_confidence(lap_var, edge_density):
    lap_score  = min(1.0, max(0.0, (lap_var - STONE_LAP_MIN) / (STONE_LAP_MIN * 0.8)))
    edge_score = min(1.0, max(0.0, (edge_density - STONE_EDGE_MIN) / (STONE_EDGE_MIN * 1.0)))
    conf = (0.6 * lap_score + 0.4 * edge_score) * 100.0
    return round(conf, 2)


# ===============================
# Helpers (คงเดิม)
# ===============================
def allowed_file_ext(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT, ext


def save_upload_bytes(filename: str, file_bytes: bytes):
    ok, ext = allowed_file_ext(filename)
    if not ok:
        return None, None, "BAD_EXT"

    # ถ้าเป็น gif -> รับไฟล์ได้ แต่ให้ขึ้นข้อความเฉพาะ
    if ext == ".gif":
        return None, None, "GIF_NOT_ALLOWED"

    # กันชื่อซ้ำ + บังคับนามสกุลให้เป็นรูปทั่วไป
    ext = ext if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"] else ".jpg"

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return file_path, unique_name, "OK"


# ===============================
# 🧠 AI Predict (คงเดิม)
# ===============================
def _predict_probs(pil_img: Image.Image, model, transform, crack_idx, no_crack_idx):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return float(probs[crack_idx].item()), float(probs[no_crack_idx].item())


def predict_image_ai(pil_img: Image.Image, model, transform, crack_idx, no_crack_idx):
    if not USE_MULTI_CROP:
        c, n = _predict_probs(pil_img, model, transform, crack_idx, no_crack_idx)
        return c, n, [c]

    W, H = pil_img.size
    crop_size = int(min(W, H) * CROP_RATIO)
    crop_size = max(32, crop_size)

    def crop_box(x, y):
        return (x, y, x + crop_size, y + crop_size)

    boxes = [
        crop_box(0, 0),
        crop_box(W - crop_size, 0),
        crop_box(0, H - crop_size),
        crop_box(W - crop_size, H - crop_size),
        crop_box((W - crop_size)//2, (H - crop_size)//2),
    ]

    if USE_9_CROP:
        boxes += [
            crop_box((W - crop_size)//2, 0),
            crop_box((W - crop_size)//2, H - crop_size),
            crop_box(0, (H - crop_size)//2),
            crop_box(W - crop_size, (H - crop_size)//2),
        ]

    crack_probs = []
    no_probs = []
    for b in boxes:
        patch = pil_img.crop(b)
        c, n = _predict_probs(patch, model, transform, crack_idx, no_crack_idx)
        crack_probs.append(c)
        no_probs.append(n)

    return max(crack_probs), max(no_probs), crack_probs


def decide_crack(crack_max, crack_probs):
    crack_hits = sum(p >= HIT_THRESHOLD for p in crack_probs)
    is_crack = (crack_max >= CRACK_THRESHOLD) or (crack_hits >= HIT_K)
    return is_crack, crack_hits


# ===============================
# Streamlit UI (แต่งเว็บอย่างเดียว)
# ===============================
st.set_page_config(page_title="Stone AI Inspection", layout="wide")

# ซ่อนส่วนของ Streamlit ให้เหมือนเว็บ
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 0.65rem; padding-bottom: 2.5rem; max-width: 1100px;}
    </style>
    """,
    unsafe_allow_html=True
)

# CSS ธีมแบบ HTML เดิม
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    *{box-sizing:border-box;}

    :root{
      --bg:#0b1423;
      --card:rgba(255,255,255,0.05);
      --text:white;
      --accent1:#00bfff;
      --accent2:#00ffcc;
      --success:#00e676;
      --danger:#ff5252;
      --warning:#ff9800;
    }
    body.light{
      --bg:#f4f7fb;
      --card:rgba(0,0,0,0.05);
      --text:#111827;
      --accent1:#2563eb;
      --accent2:#06b6d4;
      --success:#16a34a;
      --danger:#dc2626;
      --warning:#f59e0b;
    }

    html, body, [data-testid="stAppViewContainer"]{
      background: var(--bg) !important;
      color: var(--text) !important;
      font-family: 'Inter', sans-serif !important;
      transition: 0.4s ease;
    }

    /* grid background */
    [data-testid="stAppViewContainer"]::before{
      content:"";
      position:fixed;
      inset:0;
      background-image:
        linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
      background-size:40px 40px;
      animation:moveGrid 25s linear infinite;
      z-index:-2;
      pointer-events:none;
    }
    @keyframes moveGrid{
      from{background-position:0 0;}
      to{background-position:120px 120px;}
    }

    .topbar{
      display:flex;
      justify-content:space-between;
      align-items:center;
      padding:18px 22px;
      background:var(--card);
      backdrop-filter:blur(15px);
      border-radius:14px;
      margin-top:12px;
    }
    .brand{font-weight:800;}
    .status{display:flex;align-items:center;gap:8px;font-size:13px;}
    .dot{
      width:8px;height:8px;border-radius:50%;
      background:var(--accent2);
      box-shadow:0 0 10px var(--accent2);
      animation:pulse 2s infinite;
    }
    @keyframes pulse{0%{opacity:.5}50%{opacity:1}100%{opacity:.5}}

    .theme-toggle{
      cursor:pointer;
      padding:8px 18px;
      border-radius:20px;
      border:1px solid var(--accent1);
      transition:0.3s;
      user-select:none;
      font-size:13px;
    }
    .theme-toggle:hover{background:var(--accent1);color:white;}

    .containerCard{
      margin: 26px auto 0 auto;
      padding: 42px 34px;
      border-radius: 25px;
      background: var(--card);
      backdrop-filter: blur(25px);
      box-shadow: 0 0 40px rgba(0,255,255,0.05);
    }

    .title{
      text-align:center;
      font-family:'Orbitron',sans-serif;
      font-size:42px;
    }
    .ai{
      background:linear-gradient(270deg,var(--accent1),var(--accent2),#8b5cf6,var(--accent1));
      background-size:600% 600%;
      -webkit-background-clip:text;
      -webkit-text-fill-color:transparent;
      animation:gradientFlow 6s ease infinite;
    }
    @keyframes gradientFlow{
      0%{background-position:0% 50%;}
      50%{background-position:100% 50%;}
      100%{background-position:0% 50%;}
    }

    .subtitle{
      text-align:center;
      margin:15px auto 25px auto;
      font-size:16px;
      opacity:0.85;
      max-width:700px;
    }

    /* buttons */
    .stButton>button{
      padding:12px 25px !important;
      border:none !important;
      border-radius:10px !important;
      background:linear-gradient(90deg,var(--accent1),var(--accent2)) !important;
      color:white !important;
      transition:0.3s !important;
      width: 100%;
    }
    .stButton>button:hover{
      transform:scale(1.03);
      box-shadow:0 0 15px var(--accent1);
    }

    /* file uploader */
    [data-testid="stFileUploaderDropzone"]{
      border:2px dashed var(--accent1) !important;
      border-radius:12px !important;
      background:transparent !important;
      padding: 18px !important;
    }
    [data-testid="stFileUploaderDropzone"] *{color:var(--text) !important;}
    [data-testid="stFileUploaderDropzone"] svg {opacity: 0.9;}

    /* badge */
    .badge{
      padding:20px 35px;
      border-radius:50px;
      margin-top:20px;
      display:inline-block;
      font-weight:600;
      text-align:center;
    }
    .success{border:1px solid var(--success);color:var(--success);}
    .danger{border:1px solid var(--danger);color:var(--danger);}
    .warning{border:1px solid var(--warning);color:var(--warning);}

    /* progress */
    .progressWrap{
      width:350px;height:8px;
      background:rgba(255,255,255,0.1);
      margin:20px auto;
      border-radius:20px;
      overflow:hidden;
    }
    .bar{
      height:100%;
      border-radius:20px;
      width:0%;
      transition:width 1.2s ease;
    }

    /* images */
    .image-grid{
      display:flex;
      gap:40px;
      justify-content:center;
      margin-top:35px;
      flex-wrap:wrap;
    }
    .image-box{
      background:var(--card);
      padding:25px;
      border-radius:20px;
      position:relative;
      overflow:hidden;
      transition:all 0.35s ease;
      border:1px solid rgba(255,255,255,0.05);
      max-width: 410px;
    }
    .image-box:hover{
      transform:translateY(-10px);
      box-shadow:0 25px 45px rgba(0,0,0,0.25);
    }
    .image-title{
      margin: 0 0 12px 0;
      font-size: 18px;
      font-weight: 700;
    }
    .image-box img{
      width:340px;
      border-radius:15px;
      transition:0.4s;
      display:block;
      margin: 0 auto;
    }
    .image-box:hover img{transform:scale(1.06);}

    /* AI panel */
    .ai-panel{
      margin-top:35px;
      display:flex;
      justify-content:center;
      gap:25px;
      flex-wrap:wrap;
    }
    .ai-card{
      background:var(--card);
      padding:18px 25px;
      border-radius:15px;
      min-width:160px;
      text-align:center;
      backdrop-filter:blur(10px);
      border:1px solid rgba(255,255,255,0.05);
      transition:0.3s;
    }
    .ai-card:hover{
      transform:translateY(-5px);
      box-shadow:0 10px 25px rgba(0,255,255,0.1);
    }
    .ai-label{font-size:12px;opacity:0.6;margin-bottom:8px;}
    .ai-value{font-size:22px;font-weight:600;}

    .footer{
      text-align:center;
      margin-top:45px;
      opacity:0.6;
      font-size:13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Topbar + theme toggle + scan sound
components.html(
    """
    <audio id="scanSound" src="https://assets.mixkit.co/active_storage/sfx/2571/2571-preview.mp3"></audio>

    <div class="topbar">
      <div class="brand"><b>Stone <span class="ai">AI</span> Inspection</b></div>
      <div style="display:flex;gap:20px;align-items:center;">
        <div class="status"><div class="dot" id="statusDot"></div> AI Online</div>
        <div class="theme-toggle" onclick="toggleTheme()">🌙 / ☀</div>
      </div>
    </div>

    <script>
      function applyTheme(){
        const theme = localStorage.getItem("theme");
        if(theme === "light"){ document.body.classList.add("light"); }
        else{ document.body.classList.remove("light"); }
      }
      function toggleTheme(){
        document.body.classList.toggle("light");
        localStorage.setItem("theme", document.body.classList.contains("light") ? "light" : "dark");
      }
      applyTheme();
    </script>
    """,
    height=72
)

# เปิด containerCard
st.markdown('<div class="containerCard">', unsafe_allow_html=True)

# Title / subtitle
st.markdown('<div class="title">Stone Defect Detection <span class="ai">AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ระบบตรวจสอบคุณภาพของหินด้วยเทคโนโลยี AI Vision</div>', unsafe_allow_html=True)

# โหลดโมเดล (ระบบเดิม)
with st.spinner("Loading model..."):
    model, class_to_idx, crack_idx, no_crack_idx, IMG_SIZE, transform = load_model_and_meta()

# state เก็บรูปล่าสุด (เหมือน last_uploaded_path)
if "last_uploaded_path" not in st.session_state:
    st.session_state.last_uploaded_path = None
if "last_unique_name" not in st.session_state:
    st.session_state.last_unique_name = None

with st.expander("Debug/Config", expanded=False):
    st.write("Using device:", str(DEVICE))
    st.write("class_to_idx:", class_to_idx)
    st.write("USE_MULTI_CROP:", USE_MULTI_CROP, "| USE_9_CROP:", USE_9_CROP, "| CROP_RATIO:", CROP_RATIO)
    st.write("CRACK_THRESHOLD:", CRACK_THRESHOLD, "| HIT_THRESHOLD:", HIT_THRESHOLD, "| HIT_K:", HIT_K)
    st.write("STONE_LAP_MIN:", STONE_LAP_MIN, "| STONE_EDGE_MIN:", STONE_EDGE_MIN)

# uploader + buttons
uploaded = st.file_uploader("อัปโหลดรูปหิน", type=["jpg", "jpeg", "png", "webp", "bmp", "gif"])
colL, colR = st.columns([1, 1])
with colL:
    run_btn = st.button("ตรวจสอบคุณภาพ", use_container_width=True)
with colR:
    rescan_btn = st.button("Scan อีกครั้ง", use_container_width=True)

def run_scan_from_path(file_path: str):
    start_time = time.time()

    # ---- CV gate ----
    bgr = cv2.imread(file_path)
    if bgr is None:
        return {
            "status": "BAD_IMAGE",
            "result_text": "❌ ไฟล์ภาพไม่ถูกต้อง",
            "confidence": 0,
            "crack": False,
            "crack_count": 0,
            "processing_time": round(time.time() - start_time, 3),
            "file_path": file_path,
        }

    ok_stone, lap_var, edge_density = is_stone_cv(bgr)

    if not ok_stone:
        return {
            "status": "NOT_STONE",
            "result_text": "❌ ไม่ใช่หิน",
            "confidence": stone_confidence(lap_var, edge_density),
            "crack": False,
            "crack_count": 0,
            "processing_time": round(time.time() - start_time, 3),
            "file_path": file_path,
        }

    # ---- AI crack ----
    pil_img = Image.open(file_path).convert("RGB")
    crack_max, no_crack_max, crack_probs = predict_image_ai(pil_img, model, transform, crack_idx, no_crack_idx)
    is_crack, crack_hits = decide_crack(crack_max, crack_probs)

    crack = bool(is_crack)
    confidence = round((crack_max if crack else no_crack_max) * 100, 2)
    result_text = "❌ พบรอยแตก" if crack else "✅ ไม่พบรอยแตก"

    return {
        "status": "CRACK" if crack else "NO_CRACK",
        "result_text": result_text,
        "confidence": confidence,
        "crack": crack,
        "crack_count": 1 if crack else 0,
        "processing_time": round(time.time() - start_time, 3),
        "file_path": file_path,
        "crack_hits": crack_hits,
        "crack_probs": crack_probs,
    }

result = None
original_image = None

# ---- กดตรวจสอบคุณภาพ (เหมือน POST /) ----
if run_btn:
    if uploaded is None:
        st.warning("กรุณาอัปโหลดรูปก่อน")
    else:
        file_bytes = uploaded.getvalue()
        file_path, unique_name, status = save_upload_bytes(uploaded.name, file_bytes)

        # ---- handle upload status ----
        start_time = time.time()
        processing_time = round(time.time() - start_time, 3)

        if status == "GIF_NOT_ALLOWED":
            result = {
                "status": "GIF",
                "result_text": "❌ ยังไม่รองรับไฟล์ GIF",
                "confidence": 0,
                "crack": False,
                "crack_count": 0,
                "processing_time": processing_time,
                "file_path": None,
            }
            st.session_state.last_uploaded_path = None
            st.session_state.last_unique_name = None

        elif status == "BAD_EXT" or file_path is None:
            result = {
                "status": "BAD_IMAGE",
                "result_text": "❌ ไฟล์ภาพไม่ถูกต้อง",
                "confidence": 0,
                "crack": False,
                "crack_count": 0,
                "processing_time": processing_time,
                "file_path": None,
            }
            st.session_state.last_uploaded_path = None
            st.session_state.last_unique_name = None

        else:
            st.session_state.last_uploaded_path = file_path
            st.session_state.last_unique_name = unique_name
            original_image = file_path
            result = run_scan_from_path(file_path)

# ---- กด Scan อีกครั้ง (เหมือน POST /rescan) ----
if rescan_btn:
    if not st.session_state.last_uploaded_path or not os.path.exists(st.session_state.last_uploaded_path):
        st.warning("ยังไม่มีรูปสำหรับสแกนซ้ำ")
    else:
        original_image = st.session_state.last_uploaded_path
        result = run_scan_from_path(st.session_state.last_uploaded_path)

# ---- Render result ----
if result is not None and result.get("result_text"):
    crack = bool(result.get("crack", False))
    result_text = result["result_text"]
    confidence = result.get("confidence", None)

    if result_text == "❌ ไม่ใช่หิน":
        badge_class = "warning"
    elif crack:
        badge_class = "danger"
    else:
        badge_class = "success"

    # Badge
    st.markdown(
        f'<div style="text-align:center;">'
        f'<div class="badge {badge_class}">{result_text}'
        + (f"<br>ความมั่นใจ {confidence:.2f}%" if confidence is not None else "")
        + "</div></div>",
        unsafe_allow_html=True
    )

    # Progress bar (เหมือนเว็บ)
    if confidence is not None:
        bar_color = "var(--warning)" if result_text == "❌ ไม่ใช่หิน" else ("var(--danger)" if crack else "var(--success)")
        st.markdown(
            f"""
            <div class="progressWrap">
              <div class="bar" style="background:{bar_color}; width:{confidence:.2f}%;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Images grid (2 กล่อง)
    st.markdown('<div class="image-grid">', unsafe_allow_html=True)

    # left image
    if original_image and os.path.exists(original_image):
        st.markdown(
            f"""
            <div class="image-box">
              <div class="image-title">ภาพต้นฉบับ</div>
              <img src="data:image/jpeg;base64," style="display:none;">
            </div>
            """,
            unsafe_allow_html=True
        )
        # ใช้ st.image เพื่อให้ streamlit render รูปจริง (คงระบบเดิม)
        with st.container():
            st.markdown('<div class="image-box">', unsafe_allow_html=True)
            st.markdown('<div class="image-title">ภาพต้นฉบับ</div>', unsafe_allow_html=True)
            st.image(original_image, width=340)
            st.markdown('</div>', unsafe_allow_html=True)

    # right image
    if original_image and os.path.exists(original_image):
        with st.container():
            st.markdown('<div class="image-box">', unsafe_allow_html=True)
            st.markdown('<div class="image-title">ผลการตรวจจับ</div>', unsafe_allow_html=True)
            st.image(original_image, width=340)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # AI Panel
    st.markdown('<div class="ai-panel">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ai-card">
          <div class="ai-label">Crack Count</div>
          <div class="ai-value">{result.get("crack_count", 0)}</div>
        </div>
        <div class="ai-card">
          <div class="ai-label">Processing Time</div>
          <div class="ai-value">{result.get("processing_time", 0):.3f}s</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if confidence is not None:
        st.markdown(
            f"""
            <div class="ai-card">
              <div class="ai-label">AI Confidence</div>
              <div class="ai-value">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# footer
st.markdown('<div class="footer">© 2026 Stone AI Inspection | Advanced Vision Technology</div>', unsafe_allow_html=True)

# ปิด containerCard
st.markdown('</div>', unsafe_allow_html=True)

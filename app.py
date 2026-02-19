# ============================================================
# Stone Defect Detection AI (Merged + UI Tweaks by Mon)

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
import requests


# ============================================================
# CONFIG (Logic/Thresholds)
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model download (Hugging Face) ----
MODEL_PATH = "best_model.pth"
HF_MODEL_URL = "https://huggingface.co/Mon2948/best_model/resolve/main/best_model.pth?download=true"

# ---- Upload folder ----
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- LOCKED BEST SETTINGS  ----
CRACK_THRESHOLD = 0.58  # crack_max >= 0.58 -> แตก
HIT_THRESHOLD = 0.48    # ต่อ crop
HIT_K = 2               # ต้องเจออย่างน้อย 2 crop ถึงถือว่าแตก

# ---- Multi-crop ----
USE_MULTI_CROP = True
CROP_RATIO = 0.75
USE_9_CROP = True

# ---- Stone gate (OpenCV) ----
STONE_LAP_MIN = 90.0     # 80-140
STONE_EDGE_MIN = 0.015   # 0.01-0.03

# ---- Allowed extensions ----
# NOTE: อนุญาต gif ให้ "อัปได้" แต่จะตอบว่า "ยังไม่รองรับ GIF"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stone AI Inspection",
    page_icon="🪨",
    layout="wide"
)


# ============================================================
# ULTRA PREMIUM CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at 20% 20%, #0f2027, #0b1423 60%);
    color: white;
    overflow-x: hidden;
}

/* Animated Grid Background */
body::before{
    content:"";
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
@keyframes moveGrid{
    from {transform: translate(0,0);}
    to   {transform: translate(-60px,-60px);}
}

/* Header Title */
.title{
    font-family:'Orbitron', sans-serif;
    font-size: 52px;
    text-align:center;
    background: linear-gradient(270deg,#00f5ff,#00ffcc,#8b5cf6,#00f5ff);
    background-size:600% 600%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: flow 6s ease infinite;
    margin-top:18px;
    margin-bottom:0px;
}
@keyframes flow{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

.subtitle{
    text-align:center;
    opacity:0.75;
    margin-bottom:26px;
}

/* Online Dot */
.online-dot{
    height:10px;
    width:10px;
    background:#00ff95;
    border-radius:50%;
    display:inline-block;
    box-shadow:0 0 10px #00ff95;
    margin-right:8px;
    animation:pulse 2s infinite;
}
@keyframes pulse{
    0%{box-shadow:0 0 5px #00ff95;}
    50%{box-shadow:0 0 20px #00ff95;}
    100%{box-shadow:0 0 5px #00ff95;}
}

/* Glass Card */
.glass{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    padding: 30px 30px 26px 30px;
    border-radius: 25px;
    border: 1px solid rgba(0,255,255,0.12);
    box-shadow: 0 0 60px rgba(0,255,255,0.08);
    animation: fadeIn 1.2s ease;
}
@keyframes fadeIn{
    from{opacity:0; transform:translateY(16px);}
    to{opacity:1; transform:translateY(0);}
}

/* Button Glow */
.stButton>button{
    background: linear-gradient(90deg,#00f5ff,#00ffcc);
    color: black;
    border:none;
    padding: 12px 28px;
    border-radius: 14px;
    font-weight: 700;
    transition: 0.25s;
}
.stButton>button:hover{
    transform: scale(1.04);
    box-shadow: 0 0 25px rgba(0,255,204,0.9);
}

/* File uploader tweaks */
section[data-testid="stFileUploaderDropzone"]{
    border-radius:16px;
    border:1px dashed rgba(0,255,255,0.25);
    background: rgba(255,255,255,0.03);
}

/* Result ring */
.result-ring-wrap{
    display:flex;
    align-items:center;
    justify-content:center;
    margin: 10px 0 8px 0;
}
.result-ring{
    width: 190px;
    height: 190px;
    border-radius: 50%;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 0.2px;
    background:
        radial-gradient(circle, rgba(11,20,35,1) 45%, transparent 46%),
        conic-gradient(var(--ring-color) var(--percent), rgba(30,41,59,1) var(--percent));
    box-shadow: 0 0 45px rgba(0,255,204,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}

/* Badge */
.badge{
    display:inline-block;
    padding: 12px 18px;
    border-radius: 999px;
    font-weight: 800;
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.05);
    margin-bottom: 10px;
}
.badge.ok{
    color: #00ffcc;
    border-color: rgba(0,255,204,0.6);
}
.badge.bad{
    color: #ff4d6d;
    border-color: rgba(255,77,109,0.6);
}
.badge.warn{
    color: #fbbf24;
    border-color: rgba(251,191,36,0.7);
}

/* Metric Cards */
.metric-card{
    flex: 1 1 180px;
    min-width: 180px;
    background: rgba(255,255,255,0.05);
    padding: 18px 18px;
    border-radius: 18px;
    text-align:center;
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.25s;
}
.metric-card:hover{
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0,255,255,0.12);
}
.metric-title{
    opacity: 0.72;
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 6px;
}
.metric-value{
    font-size: 30px;
    font-weight: 900;
}

/* Image panels */
.panel-title{
    font-weight: 800;
    letter-spacing: 0.2px;
    margin-bottom: 10px;
}
.hr{
    height:1px;
    background: rgba(255,255,255,0.08);
    margin: 14px 0;
}

/* Footer */
.footer{
    text-align:center;
    margin-top: 30px;
    opacity: 0.45;
    font-size: 13px;
    padding-bottom: 8px;
}

/* Small helper */
.small-muted{
    opacity:0.72;
    font-size: 13px;
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    st.info("📥 Downloading model from Hugging Face...")
    r = requests.get(HF_MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


# ============================================================
# LOAD MODEL (checkpoint dict) - cache
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_and_meta():
    ensure_model()

    # ให้ torch load numpy scalar ได้เหมือนของเดิม
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


# ============================================================
# CV Stone Gate
# ============================================================

def is_stone_cv(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / (bgr_img.shape[0] * bgr_img.shape[1]))

    is_stone = (lap_var >= STONE_LAP_MIN) and (edge_density >= STONE_EDGE_MIN)
    return is_stone, float(lap_var), float(edge_density)

def stone_confidence(lap_var, edge_density):
    lap_score = min(1.0, max(0.0, (lap_var - STONE_LAP_MIN) / (STONE_LAP_MIN * 0.8)))
    edge_score = min(1.0, max(0.0, (edge_density - STONE_EDGE_MIN) / (STONE_EDGE_MIN * 1.0)))
    conf = (0.6 * lap_score + 0.4 * edge_score) * 100.0
    return round(conf, 2)


# ============================================================
# Upload helpers
# ============================================================

def allowed_file_ext(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT, ext

def save_upload_bytes(filename: str, file_bytes: bytes):
    ok, ext = allowed_file_ext(filename)
    if not ok:
        return None, None, "BAD_EXT"

    # gif -> รับไฟล์ได้ แต่ตอบข้อความเฉพาะ
    if ext == ".gif":
        return None, None, "GIF_NOT_ALLOWED"

    # กันชื่อซ้ำ + บังคับนามสกุลให้เป็นรูปทั่วไป
    ext = ext if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"] else ".jpg"

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return file_path, unique_name, "OK"


# ============================================================
# AI Predict
# ============================================================

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
        crop_box((W - crop_size) // 2, (H - crop_size) // 2),
    ]

    if USE_9_CROP:
        boxes += [
            crop_box((W - crop_size) // 2, 0),
            crop_box((W - crop_size) // 2, H - crop_size),
            crop_box(0, (H - crop_size) // 2),
            crop_box(W - crop_size, (H - crop_size) // 2),
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


# ============================================================
# Scan runner (รวม CV + AI)
# ============================================================

def run_scan_from_path(file_path: str, model, transform, crack_idx, no_crack_idx):
    start_time = time.time()

    # ---- Load bgr ----
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
            "lap_var": None,
            "edge_density": None,
            "stone_conf": 0,
            "crack_hits": 0,
            "crack_probs": [],
            "crack_max": 0,
            "no_crack_max": 0,
        }

    # ---- CV gate ----
    ok_stone, lap_var, edge_density = is_stone_cv(bgr)
    sconf = stone_confidence(lap_var, edge_density)

    if not ok_stone:
        return {
            "status": "NOT_STONE",
            "result_text": "❌ ไม่ใช่หิน",
            "confidence": sconf,
            "crack": False,
            "crack_count": 0,
            "processing_time": round(time.time() - start_time, 3),
            "file_path": file_path,
            "lap_var": lap_var,
            "edge_density": edge_density,
            "stone_conf": sconf,
            "crack_hits": 0,
            "crack_probs": [],
            "crack_max": 0,
            "no_crack_max": 0,
        }

    # ---- AI crack ----
    pil_img = Image.open(file_path).convert("RGB")
    crack_max, no_crack_max, crack_probs = predict_image_ai(
        pil_img, model, transform, crack_idx, no_crack_idx
    )
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
        "lap_var": lap_var,
        "edge_density": edge_density,
        "stone_conf": sconf,
        "crack_hits": crack_hits,
        "crack_probs": crack_probs,
        "crack_max": crack_max,
        "no_crack_max": no_crack_max,
    }


# ============================================================
# UI helper: ring + badge + metrics
# ============================================================

def ring_html(confidence: float, status: str):
    if status == "CRACK":
        ring_color = "rgba(255,77,109,0.95)"
    elif status in ("NOT_STONE", "GIF", "BAD_IMAGE"):
        ring_color = "rgba(251,191,36,0.95)"
    else:
        ring_color = "rgba(0,255,204,0.95)"

    percent = int(max(0, min(100, round(confidence))))
    return f"""
    <div class="result-ring-wrap">
      <div class="result-ring" style="--percent:{percent}%; --ring-color:{ring_color};">
        {confidence:.2f}%
      </div>
    </div>
    """

def badge_html(result_text: str, status: str):
    if status == "CRACK":
        cls = "bad"
    elif status in ("NOT_STONE", "GIF", "BAD_IMAGE"):
        cls = "warn"
    else:
        cls = "ok"
    return f'<div class="badge {cls}">{result_text}</div>'

def metric_card(title: str, value: str):
    return f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
    </div>
    """


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="title">Stone Defect Detection AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle"><span class="online-dot"></span>AI System Online • ระบบตรวจสอบคุณภาพของหินด้วย AI Vision</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("🧠 Loading model..."):
    model, class_to_idx, crack_idx, no_crack_idx, IMG_SIZE, transform = load_model_and_meta()


# ============================================================
# SESSION STATE (เก็บรูปไว้ rescan)
# ============================================================

if "last_uploaded_path" not in st.session_state:
    st.session_state.last_uploaded_path = None
if "last_unique_name" not in st.session_state:
    st.session_state.last_unique_name = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# SIDEBAR (Debug/Config)
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    show_debug = st.toggle("Show Debug/Config", value=False)

    st.markdown("---")
    st.markdown("### 🔒 Locked Settings")
    st.write("CRACK_THRESHOLD:", CRACK_THRESHOLD)
    st.write("HIT_THRESHOLD:", HIT_THRESHOLD)
    st.write("HIT_K:", HIT_K)
    st.write("USE_MULTI_CROP:", USE_MULTI_CROP)
    st.write("USE_9_CROP:", USE_9_CROP)
    st.write("CROP_RATIO:", CROP_RATIO)

    st.markdown("---")
    st.markdown("### 🧱 Stone Gate")
    st.write("STONE_LAP_MIN:", STONE_LAP_MIN)
    st.write("STONE_EDGE_MIN:", STONE_EDGE_MIN)

    st.markdown("---")
    st.markdown("### 💻 Runtime")
    st.write("Device:", str(DEVICE))
    if show_debug:
        st.write("class_to_idx:", class_to_idx)
        st.write("IMG_SIZE:", IMG_SIZE)


# ============================================================
# MAIN GLASS CONTAINER
# ============================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

# --- Upload + Buttons row
# ============================================================
# Upload Section 

center_col = st.columns([1, 2, 1])[1]

with center_col:
    uploaded = st.file_uploader(
        "อัปโหลดรูปหิน (รองรับ: jpg / png / webp / bmp / gif*)",
        type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
        help="*GIF อัปได้ แต่ระบบจะตอบว่ายังไม่รองรับตามสเปคที่ตั้งไว้"
    )

    st.markdown(
        '<div class="small-muted" style="text-align:center;">Tip: รูปชัด แสงพอดี จะผ่าน Stone Gate ง่ายขึ้น</div>',
        unsafe_allow_html=True
    )

    run_btn = st.button(" ตรวจสอบคุณภาพ", use_container_width=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)



# ============================================================
# SCAN LOGIC
# ============================================================

result = None
original_image_path = None

if run_btn:
    if uploaded is None:
        st.warning("กรุณาอัปโหลดรูปก่อน")
    else:
        file_bytes = uploaded.getvalue()
        file_path, unique_name, status = save_upload_bytes(uploaded.name, file_bytes)

        # upload handling timing
        t0 = time.time()
        processing_time = round(time.time() - t0, 3)

        if status == "GIF_NOT_ALLOWED":
            result = {
                "status": "GIF",
                "result_text": "❌ ยังไม่รองรับไฟล์ GIF",
                "confidence": 0,
                "crack": False,
                "crack_count": 0,
                "processing_time": processing_time,
                "file_path": None,
                "lap_var": None,
                "edge_density": None,
                "stone_conf": 0,
                "crack_hits": 0,
                "crack_probs": [],
                "crack_max": 0,
                "no_crack_max": 0,
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
                "lap_var": None,
                "edge_density": None,
                "stone_conf": 0,
                "crack_hits": 0,
                "crack_probs": [],
                "crack_max": 0,
                "no_crack_max": 0,
            }
            st.session_state.last_uploaded_path = None
            st.session_state.last_unique_name = None

        else:
            st.session_state.last_uploaded_path = file_path
            st.session_state.last_unique_name = unique_name
            original_image_path = file_path

            with st.spinner("🔍 AI Scanning..."):
                result = run_scan_from_path(file_path, model, transform, crack_idx, no_crack_idx)


# persist last result
if result is not None:
    st.session_state.last_result = result


# ============================================================
# RENDER RESULT (Premium Layout)
# ============================================================

to_show = st.session_state.last_result

if to_show is not None:
    status = to_show.get("status", "")
    result_text = to_show.get("result_text", "")
    confidence = float(to_show.get("confidence", 0) or 0)
    crack = bool(to_show.get("crack", False))
    ptime = float(to_show.get("processing_time", 0) or 0)

    # Debug values (only show if debug enabled)
    lap_var = to_show.get("lap_var", None)
    edge_density = to_show.get("edge_density", None)
    crack_hits = int(to_show.get("crack_hits", 0) or 0)

    # Top row: Left image + Right status
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="panel-title"> ภาพต้นฉบับ</div>', unsafe_allow_html=True)

        img_path = None
        if original_image_path and os.path.exists(original_image_path):
            img_path = original_image_path
        elif st.session_state.last_uploaded_path and os.path.exists(st.session_state.last_uploaded_path):
            img_path = st.session_state.last_uploaded_path

        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.info("ยังไม่มีภาพให้แสดง (หรือเป็น GIF/ไฟล์ไม่ถูกต้อง)")

    with right:
        st.markdown('<div class="panel-title"> ผลการตรวจจับ</div>', unsafe_allow_html=True)

    # 1) ป้ายผล (badge)
        st.markdown(badge_html(result_text, status), unsafe_allow_html=True)

    # 2) bar สีฟ้า
        st.progress(max(0.0, min(1.0, confidence / 100.0)))

    # 3) กล่องข้อความสรุป
        if status == "CRACK":
            st.error("ตรวจพบรอยแตก")
        elif status == "NO_CRACK":
            st.success("ไม่พบรอยแตก")
        elif status == "NOT_STONE":
            st.warning("ภาพไม่ผ่าน Stone Gate: ระบบมองว่าไม่ใช่หิน")
        elif status == "GIF":
            st.warning("GIF ยังไม่รองรับ — เปลี่ยนเป็น JPG/PNG/WebP ก่อน")
        else:
            st.warning("ไม่สามารถประมวลผลภาพได้")

    
        st.markdown(ring_html(confidence, status), unsafe_allow_html=True)



    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    
    m1, m2, m3 = st.columns([1, 1, 1])

    with m1:
        st.markdown(metric_card("Crack Count", str(to_show.get("crack_count", 0))), unsafe_allow_html=True)

    with m2:
        st.markdown(metric_card("AI Confidence", f"{confidence:.2f}%"), unsafe_allow_html=True)

    with m3:
        st.markdown(metric_card("Processing Time", f"{ptime:.3f}s"), unsafe_allow_html=True)

    # Optional Debug details
    if show_debug:
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("###  Debug Details")

        cA, cB = st.columns([1, 1])
        with cA:
            st.write("status:", status)
            st.write("crack:", crack)
            st.write("crack_max:", float(to_show.get("crack_max", 0)))
            st.write("no_crack_max:", float(to_show.get("no_crack_max", 0)))
            st.write("crack_hits:", crack_hits)

        with cB:
            st.write("lap_var:", lap_var)
            st.write("edge_density:", edge_density)
            st.write("STONE_LAP_MIN:", STONE_LAP_MIN)
            st.write("STONE_EDGE_MIN:", STONE_EDGE_MIN)

        probs = to_show.get("crack_probs", []) or []
        if probs:
            st.write("crack_probs (per crop):")
            st.write([round(float(p), 4) for p in probs])


# ============================================================
# CLOSE GLASS DIV
# ============================================================

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown('<div class="footer">© 2026 Stone AI Inspection | AI Vision Technology</div>', unsafe_allow_html=True)








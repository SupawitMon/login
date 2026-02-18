import streamlit as st
import base64
from pathlib import Path
import time
import random

st.set_page_config(
    page_title="Stone AI Inspection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# 🔥 BACKGROUND SYSTEM
# =========================

def set_bg_image(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def set_gradient_bg():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    </style>
    """, unsafe_allow_html=True)


# 🔁 เปลี่ยนโหมดพื้นหลัง
bg_mode = st.sidebar.selectbox("🎨 Background Mode", ["Animated Gradient", "Image"])

if bg_mode == "Animated Gradient":
    set_gradient_bg()
else:
    set_bg_image("background.jpg")  # ใส่ไฟล์เอง


# =========================
# 💎 GLOBAL STYLES
# =========================

st.markdown("""
<style>

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 0 40px rgba(0,255,255,0.2);
}

/* Title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2px;
}

.ai-text {
    color: #00f5ff;
}

/* Glow Button */
.stButton>button {
    background: linear-gradient(90deg, #00f5ff, #00ff95);
    color: black;
    border-radius: 12px;
    font-weight: bold;
    padding: 12px 25px;
    box-shadow: 0 0 20px #00f5ff;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0 0 35px #00ff95;
    transform: scale(1.05);
}

/* Result Circle */
.result-circle {
    border-radius: 50%;
    padding: 25px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: white;
    background: radial-gradient(circle, #00ff95 0%, #008b8b 100%);
    box-shadow: 0 0 40px #00ff95;
    width: 200px;
    margin: auto;
}

</style>
""", unsafe_allow_html=True)


# =========================
# 🚀 HEADER
# =========================

st.markdown(
    '<div class="main-title">Stone Defect Detection <span class="ai-text">AI</span></div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# 📤 UPLOAD
# =========================

uploaded_file = st.file_uploader("Upload Stone Image", type=["jpg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("AI Analysis")

        if st.button("🔍 Analyze Image"):
            with st.spinner("AI Processing..."):
                time.sleep(2)

                confidence = round(random.uniform(65, 95), 2)

                if confidence > 80:
                    result_text = "No Crack Detected"
                else:
                    result_text = "Crack Found"

                st.markdown(
                    f"""
                    <div class="result-circle">
                        {result_text}<br>
                        {confidence}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(confidence/100)

        st.markdown('</div>', unsafe_allow_html=True)


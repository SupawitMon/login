import streamlit as st
import time
import random

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Stone AI Inspection",
    page_icon="🪨",
    layout="wide"
)

# ===============================
# CUSTOM CSS (ULTRA PREMIUM UI)
# ===============================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(-45deg, #0b1423, #0f1b2e, #0b1423, #121c30);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 0 40px rgba(0,255,255,0.08);
}

.title {
    font-family: 'Orbitron', sans-serif;
    font-size: 48px;
    text-align:center;
    background: linear-gradient(270deg,#00bfff,#00ffcc,#8b5cf6,#00bfff);
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

.subtitle {
    text-align:center;
    opacity:0.8;
    margin-bottom:40px;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    padding:25px;
    border-radius:15px;
    text-align:center;
    transition:0.3s;
    border:1px solid rgba(255,255,255,0.08);
}

.metric-card:hover{
    transform: translateY(-8px);
    box-shadow: 0 10px 30px rgba(0,255,255,0.15);
}

.metric-label {
    font-size:14px;
    opacity:0.6;
}

.metric-value {
    font-size:28px;
    font-weight:600;
}

.stButton>button {
    background: linear-gradient(90deg,#00bfff,#00ffcc);
    color:white;
    border:none;
    padding:12px 30px;
    border-radius:10px;
    font-weight:600;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.05);
    box-shadow:0 0 20px #00ffcc;
}

.progress-bar {
    height:12px;
    border-radius:20px;
    background:rgba(255,255,255,0.1);
    overflow:hidden;
    margin-top:20px;
}

.progress-fill {
    height:100%;
    border-radius:20px;
    transition: width 1.5s ease;
}

.success { background:linear-gradient(90deg,#00e676,#00ffcc); }
.danger { background:linear-gradient(90deg,#ff5252,#ff1744); }

.footer {
    text-align:center;
    margin-top:60px;
    opacity:0.5;
    font-size:13px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown('<div class="title">Stone Defect Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Vision Inspection System</div>', unsafe_allow_html=True)

# ===============================
# MAIN GLASS CONTAINER
# ===============================
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Stone Image", type=["jpg","png","jpeg"])

    if uploaded:

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(uploaded, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.write("### 🔍 AI Processing...")
            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)

            # ====== Fake AI result (replace with your model) ======
            crack = random.choice([True, False])
            confidence = round(random.uniform(88,99.9),2)
            crack_count = random.randint(0,5)
            processing_time = round(random.uniform(0.4,1.2),2)

            st.markdown("##")

            if crack:
                st.error(f"❌ Crack Detected ({confidence}%)")
                bar_class = "danger"
            else:
                st.success(f"✅ No Crack Detected ({confidence}%)")
                bar_class = "success"

            st.markdown(f"""
            <div class="progress-bar">
                <div class="progress-fill {bar_class}" style="width:{confidence}%"></div>
            </div>
            """, unsafe_allow_html=True)

        # ===== METRIC DASHBOARD =====
        st.markdown("##")
        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Crack Count</div>
                <div class="metric-value">{crack_count}</div>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AI Confidence</div>
                <div class="metric-value">{confidence}%</div>
            </div>
            """, unsafe_allow_html=True)

        with colC:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Processing Time</div>
                <div class="metric-value">{processing_time}s</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown('<div class="footer">© 2026 Stone AI Inspection | AI Vision Technology</div>', unsafe_allow_html=True)

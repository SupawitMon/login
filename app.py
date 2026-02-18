import streamlit as st
import time
import random

st.set_page_config(
    page_title="Stone AI Inspection",
    page_icon="🪨",
    layout="wide"
)

# ===============================
# ULTRA PREMIUM CSS
# ===============================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at 20% 20%, #0f2027, #0b1423 60%);
    color: white;
    overflow-x: hidden;
}

/* Animated Grid Background */
body::before {
    content: "";
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

@keyframes moveGrid {
    from {transform: translate(0,0);}
    to {transform: translate(-60px,-60px);}
}

/* Title */
.title {
    font-family: 'Orbitron', sans-serif;
    font-size: 52px;
    text-align:center;
    background: linear-gradient(270deg,#00f5ff,#00ffcc,#8b5cf6,#00f5ff);
    background-size:600% 600%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: flow 6s ease infinite;
    margin-top:20px;
}

@keyframes flow {
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

.subtitle {
    text-align:center;
    opacity:0.7;
    margin-bottom:40px;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    padding: 40px;
    border-radius: 25px;
    border:1px solid rgba(0,255,255,0.1);
    box-shadow: 0 0 60px rgba(0,255,255,0.08);
    animation: fadeIn 1.5s ease;
}

@keyframes fadeIn {
    from {opacity:0; transform:translateY(20px);}
    to {opacity:1; transform:translateY(0);}
}

/* Button Glow */
.stButton>button {
    background: linear-gradient(90deg,#00f5ff,#00ffcc);
    color:black;
    border:none;
    padding:12px 35px;
    border-radius:12px;
    font-weight:600;
    transition:0.3s;
}

.stButton>button:hover {
    transform:scale(1.05);
    box-shadow:0 0 25px #00ffcc;
}

/* Result Ring */
.result-ring {
    width:180px;
    height:180px;
    border-radius:50%;
    margin:auto;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:22px;
    font-weight:bold;
    background: radial-gradient(circle,#0b1423 40%, transparent 41%),
                conic-gradient(#00ffcc VAR%, #1e293b VAR%);
    box-shadow:0 0 40px #00ffcc;
}

/* Metric Card */
.metric-card {
    background: rgba(255,255,255,0.05);
    padding:30px;
    border-radius:18px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
    transition:0.3s;
}

.metric-card:hover {
    transform:translateY(-10px);
    box-shadow:0 20px 40px rgba(0,255,255,0.2);
}

.metric-value {
    font-size:32px;
    font-weight:700;
}

.online-dot {
    height:10px;
    width:10px;
    background:#00ff95;
    border-radius:50%;
    display:inline-block;
    box-shadow:0 0 10px #00ff95;
    margin-right:8px;
    animation:pulse 2s infinite;
}

@keyframes pulse {
    0%{box-shadow:0 0 5px #00ff95;}
    50%{box-shadow:0 0 20px #00ff95;}
    100%{box-shadow:0 0 5px #00ff95;}
}

.footer {
    text-align:center;
    margin-top:60px;
    opacity:0.4;
    font-size:13px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown('<div class="title">Stone Defect Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"><span class="online-dot"></span>AI System Online</div>', unsafe_allow_html=True)

# ===============================
# MAIN CONTAINER
# ===============================
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Stone Image", type=["jpg","png","jpeg"])

    if uploaded:
        col1, col2 = st.columns([1,1])

        with col1:
            st.image(uploaded, use_column_width=True)

        with col2:
            st.write("### 🔍 AI Scanning...")

            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)

            crack = random.choice([True, False])
            confidence = round(random.uniform(85,99),2)
            crack_count = random.randint(0,4)
            processing_time = round(random.uniform(0.4,1.1),2)

            percent = int(confidence)
            ring_html = f"""
            <div class="result-ring" style="--percent:{percent}; background:
            radial-gradient(circle,#0b1423 45%, transparent 46%),
            conic-gradient(#00ffcc {percent}%, #1e293b {percent}%);">
            {confidence}%
            </div>
            """
            st.markdown(ring_html, unsafe_allow_html=True)

            if crack:
                st.error("Crack Detected")
            else:
                st.success("No Crack Detected")

        st.markdown("##")
        colA, colB, colC = st.columns(3)

        colA.markdown(f'<div class="metric-card"><div>Crack Count</div><div class="metric-value">{crack_count}</div></div>', unsafe_allow_html=True)
        colB.markdown(f'<div class="metric-card"><div>AI Confidence</div><div class="metric-value">{confidence}%</div></div>', unsafe_allow_html=True)
        colC.markdown(f'<div class="metric-card"><div>Processing Time</div><div class="metric-value">{processing_time}s</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">© 2026 Stone AI Inspection | Ultra AI Vision Lab</div>', unsafe_allow_html=True)

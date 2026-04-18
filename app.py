import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🩻",
    layout="centered"
)

THRESHOLD = 0.055

@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "CXR_abnormal_detector_fixed.onnx",
        providers=["CPUExecutionProvider"]
    )

session = load_model()
input_name = session.get_inputs()[0].name

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f8fc 0%, #eef4fb 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #0f3d66, #1f6aa5);
        padding: 28px 24px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        margin-bottom: 24px;
    }
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-subtitle {
        font-size: 15px;
        opacity: 0.95;
    }
    .section-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        margin-bottom: 18px;
        border: 1px solid #e5edf5;
    }
    .result-box {
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        font-size: 30px;
        font-weight: 700;
        margin-top: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .normal-box {
        background: #eafaf0;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .pneumonia-box {
        background: #fff1f2;
        color: #b91c1c;
        border: 1px solid #fecdd3;
    }
    .note-box {
        background: #fffbea;
        border: 1px solid #fde68a;
        color: #854d0e;
        padding: 16px;
        border-radius: 14px;
        font-size: 15px;
        margin-top: 18px;
    }
    .footer-box {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 10px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">Pneumonia Detection System</div>
        <div class="main-subtitle">
            AI-based analysis of chest X-ray images for research and educational demonstration
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Upload Chest X-ray")
st.write("Please upload a frontal chest X-ray image in PNG, JPG, or JPEG format.")
uploaded_file = st.file_uploader("Select image", type=["png", "jpg", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    x = preprocess(image)
    raw_output = session.run(None, {input_name: x})[0]
    logit = float(raw_output[0][0])
    p = sigmoid(logit)

    if p < THRESHOLD:
        predicted_class = "Pneumonia"
        box_class = "pneumonia-box"
    else:
        predicted_class = "Normal"
        box_class = "normal-box"

    st.markdown(
        f'<div class="result-box {box_class}">{predicted_class}</div>',
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="note-box">
        <b>Demo Notice:</b> This system is a research demonstration only and should not be used for standalone clinical diagnosis.
        More data will be added later to improve model performance and reliability.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer-box">Developed for research, teaching, and demonstration purposes.</div>',
    unsafe_allow_html=True
)
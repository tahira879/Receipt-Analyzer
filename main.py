import os
import base64
import streamlit as st
import cv2
import numpy as np
import json
import pandas as pd
import plotly.express as px
from io import BytesIO
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

# ----------------------
# 1. INITIALIZATION
# ----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Cyber OCR Engine", layout="wide")

# ----------------------
# 2. ADVANCED IMAGE PREPROCESSING
# ----------------------
def preprocess_image(image: Image.Image):
    img_np = np.array(image.convert('RGB'))

    # 1. Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 2. Noise Reduction
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 3. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. Gaussian Blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 5. Adaptive Threshold
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 6. Text Visibility Enhancement (Sharpening)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return {
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced,
        "blurred": blurred,
        "threshold": threshold,
        "sharpened": sharpened
    }

# ----------------------
# 3. UI STYLING
# ----------------------
st.markdown("""
    <style>
        .stApp { background-color: #f8fafc; }
        .ocr-neon-title { font-size: 50px !important; font-weight: 950; color: #001f3f; text-align:center; text-shadow:0 0 10px #3b82f6; }
        .section-header { background:#0f172a; color:white; padding:10px; border-radius:8px; margin: 20px 0; text-align:center; font-weight:bold; }
        .advice-box { 
            background: #e0f2fe; 
            border-left: 5px solid #0369a1; 
            padding: 20px; 
            border-radius: 10px; 
            color: #0c4a6e; 
            font-size: 16px; 
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="ocr-neon-title">AI-Powered Receipt Analyzer with LLM Insights</p>', unsafe_allow_html=True)

# ----------------------
# 4. INPUT METHOD
# ----------------------
input_mode = st.radio("SELECT INPUT SOURCE:", ["📂 Upload Image", "📸 Live Camera Scanner"], horizontal=True)

active_file = None
if input_mode == "📂 Upload Image":
    active_file = st.file_uploader("Upload Receipt", type=["jpg", "png", "jpeg"])
else:
    active_file = st.camera_input("Scan Receipt")

# ----------------------
# 5. MAIN PROCESSING PIPELINE
# ----------------------
if active_file:
    input_image = Image.open(active_file)

    # ----------------------
    # PHASE 1: ADVANCED CV PIPELINE
    # ----------------------
    st.markdown('<div class="section-header">🔍 PHASE 1: ADVANCED COMPUTER VISION PIPELINE</div>', unsafe_allow_html=True)

    processed = preprocess_image(input_image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(input_image, caption="Original", use_container_width=True)
        st.image(processed["gray"], caption="Grayscale", use_container_width=True)

    with col2:
        st.image(processed["denoised"], caption="Noise Reduction", use_container_width=True)
        st.image(processed["blurred"], caption="Gaussian Blur", use_container_width=True)

    with col3:
        st.image(processed["enhanced"], caption="Contrast Enhanced (CLAHE)", use_container_width=True)
        st.image(processed["threshold"], caption="Adaptive Threshold", use_container_width=True)
        st.image(processed["sharpened"], caption="Text Visibility Enhanced", use_container_width=True)

    # ----------------------
    # EXECUTE AI ANALYSIS
    # ----------------------
    if st.button("🚀 EXECUTE NEURAL ANALYSIS"):
        if not GROQ_API_KEY:
            st.error("API Key missing! Add GROQ_API_KEY to your .env file.")
        else:
            with st.spinner("Analyzing Receipt with LLM..."):
                try:
                    buffered = BytesIO()
                    input_image.save(buffered, format="JPEG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode()

                    client = Groq(api_key=GROQ_API_KEY)

                    prompt = """
                    Act as an expert auditor. Analyze the receipt image and return ONLY a JSON object:
                    {
                      "raw_text": "All extracted text",
                      "items": [{"name": "item", "price": 0.0, "qty": 1, "category": "Snacks/Dairy/Electronics/etc"}],
                      "total": 0.0,
                      "anomalies": "Highlight pricing issues or OCR errors"
                    }
                    """

                    response = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                            ]
                        }],
                        response_format={"type": "json_object"}
                    )

                    res_data = json.loads(response.choices[0].message.content)
                    df = pd.DataFrame(res_data.get("items", []))
                    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

                    # ----------------------
                    # PHASE 2: DATA DISPLAY
                    # ----------------------
                    st.markdown('<div class="section-header">📊 PHASE 2: DATA PARSING</div>', unsafe_allow_html=True)

                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.subheader("Raw OCR Text")
                        st.text_area("Extracted Text", res_data.get("raw_text", ""), height=300)

                    with c2:
                        st.subheader("Structured Items")
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # ----------------------
                    # PHASE 3: ANALYTICS
                    # ----------------------
                    st.markdown('<div class="section-header">📈 PHASE 3: SPENDING ANALYSIS</div>', unsafe_allow_html=True)

                    cat_spending = df.groupby("category")["price"].sum().reset_index()
                    total_spent = df["price"].sum()

                    fig1 = px.pie(cat_spending, values="price", names="category", hole=0.4)
                    st.plotly_chart(fig1, use_container_width=True)

                    fig2 = px.bar(cat_spending, x="category", y="price", color="category")
                    st.plotly_chart(fig2, use_container_width=True)

                    st.metric("Grand Total", f"${total_spent:,.2f}")
                    st.warning(f"Anomalies: {res_data.get('anomalies', 'None')}")

                    # ----------------------
                    # PHASE 4: FINANCIAL ADVICE
                    # ----------------------
                    st.markdown('<div class="section-header">💡 PHASE 4: AI FINANCIAL ADVICE</div>', unsafe_allow_html=True)

                    advice_prompt = f"Based on spending {res_data['items']} total ${total_spent}, give 3 money saving tips."

                    advice_res = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "You are a professional financial advisor."},
                            {"role": "user", "content": advice_prompt}
                        ]
                    )

                    st.markdown(
                        f'<div class="advice-box"><b>AI Recommendation:</b><br>{advice_res.choices[0].message.content}</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")

else:
    st.info("Awaiting Input: Please upload or scan a receipt.")

st.markdown(
    '<div style="text-align:center; padding:15px; background:#0f172a; color:white; border-radius:10px; margin-top:30px; font-weight:600;">AI-Powered Receipt Analyzer with LLM Insights | Computer Vision • AI Analytics • Secure Processing</div>',
    unsafe_allow_html=True
)

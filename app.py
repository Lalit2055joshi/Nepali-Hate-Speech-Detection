import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from utils import MODEL_PATHS, LABELS, load_model, predict

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Text Classification App", page_icon="🤖", layout="centered")

# ---- CUSTOM STYLES ----
st.markdown("""
    <style>
    .big-font { font-size:22px !important; }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 10px;
    }
    .prediction {
        font-size: 28px;
        font-weight: bold;
        color: green;
    }
    </style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", list(MODEL_PATHS.keys()))
st.sidebar.markdown("**ℹ Tip:** Try different models to compare results.")

# ---- MAIN TITLE ----
st.title("🤖 Transformer Model Inference")
st.write("Classify text into categories like **Bias**, **Hate Speech**, **Misinformation**, and **Normal**.")

# ---- LOAD MODEL ----
tokenizer, model = load_model(MODEL_PATHS[model_choice])

# ---- TEXT INPUT ----
text_input = st.text_area("✏️ Enter text to classify:", "", height=150)

# ---- PREDICT ----
if st.button("🔍 Predict", use_container_width=True):
    if text_input.strip():
        label, conf, all_probs = predict(text_input, tokenizer, model)

        # Display Prediction
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"<div class='prediction'>Prediction: {label}</div>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {conf:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Probability Bars
        st.subheader("📊 Class Probabilities")
        for lbl, p in zip(LABELS, all_probs):
            st.progress(p)
            st.write(f"**{lbl}:** {p:.2%}")

    else:
        st.warning("⚠ Please enter some text before prediction.")
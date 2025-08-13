import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# ---- CONFIG ----
MODEL_PATHS = {
    "IndicBERT": "./models/indic-bert",
    "mBERT": "./models/mbert",
    "XLM-RoBERTa": "./models/xlm-roberta"
}

LABELS = ["bias", "hate_speech", "misinformation", "normal"]

# ---- HELPER FUNCTION ----
@st.cache_resource
def load_model(model_path):
    tokenizer_path = os.path.join(model_path, "tokenizer")
    model_pth = os.path.join(model_path, "model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_pth, num_labels=4)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()
    return LABELS[predicted_class], confidence, probs.squeeze().tolist()
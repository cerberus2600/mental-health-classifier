
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "trained_model"  # adjust this to your model directory
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Define class labels (adjust if needed)
label_map = {0: "Anxiety", 1: "Depression", 2: "Normal", 3: "Suicidal"}

st.title("Mental Health Sentiment Classifier")

# User input
user_input = st.text_area("Enter a sentence related to mental health:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.nn.functional.softmax(logits, dim=1)[0][prediction].item()

        st.success(f"Predicted Sentiment: **{label_map[prediction]}**")
        st.info(f"Confidence: {confidence:.2%}")

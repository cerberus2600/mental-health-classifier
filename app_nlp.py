import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import numpy as np

st.set_page_config(page_title="Mental Health Classifier & Chatbot", layout="wide")

# Load fine-tuned BERT model
@st.cache_resource
def load_bert_model():
    model_path = "trained_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

bert_tokenizer, bert_model = load_bert_model()
label_map = {0: "Anxiety", 1: "Depression", 2: "Normal", 3: "Suicidal"}


# Sidebar page navigation
page = st.sidebar.selectbox("Choose a page", ["Text Classification", "Chatbot"])

# ------------------ Text Classification ------------------
if page == "Text Classification":
    st.title("🧠 Mental Health Sentiment Classifier")
    user_input = st.text_area("Enter a sentence related to mental health:", height=150)

    if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()
                confidence = torch.nn.functional.softmax(logits, dim=1)[0][prediction].item()

            st.success(f"Predicted Sentiment: **{label_map[prediction]}**")
            st.info(f"Confidence: {confidence:.2%}")

# ------------------ Chatbot with Memory ------------------
elif page == "Chatbot":
    import os
    from openai import OpenAI

    st.title("🧠 Mental Health Chatbot")

    # Init history + client
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.secrets["OPENROUTER_API_KEY"]
        )

    

      # Display prior chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input field
    if user_input := st.chat_input("How can I help you today?"):
        # Step 1: Save user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Step 2: Sentiment classification using BERT
        inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            prediction = torch.argmax(logits, dim=1).item()
        sentiment = label_map[prediction]

        # ✅ Step 3 REMOVED

        # Step 4: LLM response with dynamic sentiment context
        with st.chat_message("assistant"):
            try:
                client = st.session_state.llm_client
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1:free",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a helpful and compassionate mental health assistant. The user appears to be experiencing {sentiment.lower()}."
                        }
                    ] + st.session_state.chat_history,
                    stream=True
                )
                full_reply = st.write_stream(response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_reply})
            except Exception as e:
                st.error(f"❌ Error: {e}")

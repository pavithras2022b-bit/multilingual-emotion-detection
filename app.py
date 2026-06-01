import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os

@st.cache_resource
def load_model_and_artifacts():
    """Loads the mBERT model, tokenizer, and label encoder."""
    try:
        save_dir = r"C:\Users\shanm\Downloads\my_champion_model_96_8\my_champion_model_96_8"
        encoder_path = r"C:\Users\shanm\OneDrive\Documents\NLP_project\mbert_label_encoder.pkl"

        if not os.path.isdir(save_dir):
            st.error(f"Error: Cannot find the folder '{save_dir}'.")
            st.error("Please make sure you have unzipped the file and the folder exists.")
            return None, None, None

        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)

        with open(encoder_path, "rb") as f:
            le = pickle.load(f)

        return model, tokenizer, le

    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.error(f"Failed path was: {save_dir}")
        return None, None, None


st.set_page_config(page_title="Emotion Analyzer", layout="wide")
st.markdown(
    """
    <style>
        h1 {
            text-align: center;
            color: #ff6f61;
            font-size: 42px !important;
        }
        .stTextArea label {
            font-weight: bold;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("E-Commerce Review Emotion Analyzer")
st.write("### Enter a review in Tamil, or Telugu to predict its emotion below 👇")

model, tokenizer, le = load_model_and_artifacts()

if model is not None:
    text_input = st.text_area(
        "Enter your review text here:",
        height=150,
        placeholder="இந்த போன் மிகவும் நல்லது... (This phone is very good...)"
    )

    if st.button("✨ Analyze Emotion", type="primary", use_container_width=True):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("🔍 Analyzing emotion..."):
                model.to("cpu")
                model.eval()

                inputs = tokenizer(
                    text_input,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits
                pred_id = torch.argmax(logits, dim=1).item()
                predicted_label = le.classes_[pred_id]

                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                confidence = probabilities[pred_id].item()

            # --- Results Section ---
            st.success(f"### 🧠 Predicted Emotion: **{predicted_label}**")
            st.markdown(f"**Confidence:** `{confidence:.2%}`")

            # --- Confidence Scores Chart ---
            probs_df = pd.DataFrame({
                'Emotion': le.classes_,
                'Confidence': probabilities.numpy()
            }).sort_values(by="Confidence", ascending=False)

            # Format percentages as text for display
            probs_df['Percentage'] = probs_df['Confidence'].apply(lambda x: f"{x:.2%}")

            st.subheader("📊 Confidence Scores for All Emotions")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.bar_chart(probs_df.set_index("Emotion")["Confidence"])

            with col2:
                st.dataframe(
                    probs_df[['Emotion', 'Percentage']].set_index('Emotion'),
                    use_container_width=True
                )

else:
    st.error("Model files not found. Ensure 'my_champion_model_96_8' and 'mbert_label_encoder.pkl' are in the same folder as app.py.")



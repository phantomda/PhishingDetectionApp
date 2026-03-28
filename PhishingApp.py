import streamlit as st
import pickle
import numpy as np

# Optional: function to clean input text
def clean_text(text):
    return text.lower().strip()

# Load model + vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ✅ Display model version for quick verification
MODEL_VERSION = "v2"
st.markdown(f"**Model version:** {MODEL_VERSION}")

st.set_page_config(page_title="Phishing Detection Tool", layout="centered")

st.title("🔐 AI Phishing Detection Tool")
st.markdown("Detect whether an email or URL is **phishing or legitimate** using NLP.")

st.divider()

# Input
text = st.text_area("📩 Enter email or URL:", height=150)

if st.button("Analyse"):
    if text:
        # Clean + transform input
        cleaned = clean_text(text)
        text_vec = vectorizer.transform([cleaned])

        # Prediction probability
        prob = model.predict_proba(text_vec)[0][1]  # Probability of phishing

        st.divider()

        # Result with thresholds
        if prob > 0.85:
            st.error(f"⚠️ Phishing ({prob*100:.2f}%)")
        elif prob > 0.6:
            st.warning(f"⚠️ Suspicious ({prob*100:.2f}%)")
        else:
            st.success(f"✅ Legitimate ({(1-prob)*100:.2f}%)")

        # 🔍 Explainability (top words)
        st.subheader("🔍 Key Indicators")
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        indices = text_vec.nonzero()[1]

        word_scores = []
        for i in indices:
            word = feature_names[i]
            score = coefficients[i]
            word_scores.append((word, score))

        word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:10]

        for word, score in word_scores:
            if score > 0:
                st.write(f"🔴 {word} (phishing indicator)")
            else:
                st.write(f"🟢 {word} (legitimate indicator)")

    else:
        st.warning("Please enter some text to analyse.")
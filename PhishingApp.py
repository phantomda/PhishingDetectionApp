import streamlit as st
import pickle
import numpy as np

# Load model + vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.set_page_config(page_title="Phishing Detection Tool", layout="centered")

st.title("🔐 AI Phishing Detection Tool")
st.markdown("Detect whether an email or URL is **phishing or legitimate** using NLP.")

st.divider()

# Input
text = st.text_area("📩 Enter email or URL:", height=150)

if st.button("Analyse"):
    if text:
        # Transform input
        text_vec = vectorizer.transform([text])

        # Prediction
        prediction = model.predict(text_vec)[0]
        prob = model.predict_proba(text_vec)[0]

        confidence = np.max(prob) * 100

        st.divider()

        # Result
        if prediction == 1:
            st.error(f"⚠️ Phishing Detected ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ Legitimate ({confidence:.2f}% confidence)")

        # 🔍 Explainability (top words)
        st.subheader("🔍 Key Indicators")

        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        # Get non-zero indices from input
        indices = text_vec.nonzero()[1]

        word_scores = []
        for i in indices:
            word = feature_names[i]
            score = coefficients[i]
            word_scores.append((word, score))

        # Sort by importance
        word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:10]

        for word, score in word_scores:
            if score > 0:
                st.write(f"🔴 {word} (phishing indicator)")
            else:
                st.write(f"🟢 {word} (legitimate indicator)")

    else:
        st.warning("Please enter some text to analyse.")
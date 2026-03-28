# -------------------------------
# Streamlit App: Email + URL Phishing Detection
# -------------------------------

import streamlit as st
import pickle
import re

# -------------------------------
# Model version
# -------------------------------
MODEL_VERSION = "v3 (email + URL)"

# -------------------------------
# Cleaning function (matches training)
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    # Replace URLs and emails with placeholders
    text = re.sub(r'http\S+|www\S+', ' _url_ ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' _email_ ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# Load model and vectorizer
# -------------------------------
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# -------------------------------
# Streamlit configuration
# -------------------------------
st.set_page_config(page_title="Phishing Detection Tool", layout="centered")
st.title("🔐 AI Phishing Detection Tool")
st.markdown("Detect whether an email or URL is **phishing or legitimate** using NLP.")
st.markdown(f"**Model version:** {MODEL_VERSION}")
st.divider()

# -------------------------------
# Input area
# -------------------------------
text = st.text_area("📩 Enter email or URL:", height=150)

# -------------------------------
# Test mode example URLs
# -------------------------------
if st.checkbox("Test Mode Examples"):
    st.write("""
    Examples you can test:
    - http://secure-bank-login.com
    - http://verify-paypal-account.info
    - http://update-your-apple-id.com
    - http://linkedin.com/in/username
    """)

# -------------------------------
# Analyse button
# -------------------------------
if st.button("Analyse"):
    if text:
        cleaned = clean_text(text)
        text_vec = vectorizer.transform([cleaned])

        # Prediction probability
        prob = model.predict_proba(text_vec)[0][1]

        # Get feature importance for this input
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        indices = text_vec.nonzero()[1]
        word_scores = [(feature_names[i], coefficients[i]) for i in indices]
        word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:10]

        phishing_indicators = [w for w, s in word_scores if s > 0]

        # Whitelist common safe domains
        SAFE_DOMAINS = ["linkedin.com", "github.com", "python.org"]
        is_whitelisted = any(domain in text.lower() for domain in SAFE_DOMAINS)

        # -------------------------------
        # Classification logic
        # -------------------------------
        if is_whitelisted:
            st.success(f"✅ Legitimate (whitelisted domain)")
        else:
            # If URL/email has phishing indicators
            if len(phishing_indicators) > 0:
                if prob > 0.85:
                    st.error(f"⚠️ Phishing ({prob*100:.2f}%)")
                elif prob > 0.6:
                    st.warning(f"⚠️ Suspicious ({prob*100:.2f}%)")
                else:
                    st.success(f"✅ Legitimate ({(1-prob)*100:.2f}%)")
            else:
                # No indicators: rely on probability threshold
                if prob > 0.9:
                    st.error(f"⚠️ Phishing ({prob*100:.2f}%)")
                else:
                    st.success(f"✅ Legitimate ({(1-prob)*100:.2f}%)")

        # -------------------------------
        # Display top indicators
        # -------------------------------
        st.subheader("🔍 Key Indicators")
        for word, score in word_scores:
            if score > 0:
                st.write(f"🔴 {word} (phishing indicator)")
            else:
                st.write(f"🟢 {word} (legitimate indicator)")

    else:
        st.warning("Please enter some text to analyse.")
import streamlit as st
import pickle
import numpy as np
import re
from urllib.parse import urlparse

# =======================
# Optional: Function to clean email text
# =======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' _url_ ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' _email_ ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =======================
# URL Feature Extraction
# =======================
def extract_url_features(url):
    features = {}
    features['length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['has_at'] = 1 if '@' in url else 0
    # You can add more features here
    return np.array(list(features.values())).reshape(1, -1)

# =======================
# Load models
# =======================
# Email model + TF-IDF vectorizer
email_model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Optional: a separate URL model trained with URL features
# For now, we use the email_model as a combined feature model
url_model = email_model  # placeholder

st.set_page_config(page_title="Phishing Detection Tool", layout="centered")
st.title("🔐 AI Phishing Detection Tool")
st.markdown("Detect whether an email or URL is **phishing or legitimate** using ML.")

st.divider()

# Input
text = st.text_area("📩 Enter email or URL:", height=150)

if st.button("Analyse") and text:
    # Check if input is likely a URL
    parsed = urlparse(text)
    is_url = bool(parsed.scheme and parsed.netloc)

    if is_url:
        # URL feature-based prediction
        url_features = extract_url_features(text)
        prob = url_model.predict_proba(url_features)[0][1]  # phishing probability

        # Display simple indicators
        st.subheader("🔍 URL Indicators")
        st.write(f"Length: {url_features[0][0]}")
        st.write(f"Dots: {url_features[0][1]}, Hyphens: {url_features[0][2]}, Digits: {url_features[0][3]}, Contains '@': {url_features[0][4]}")
    else:
        # Email / text-based prediction
        cleaned = clean_text(text)
        text_vec = vectorizer.transform([cleaned])
        prob = email_model.predict_proba(text_vec)[0][1]

        # Explainability: top keywords
        st.subheader("🔍 Key Indicators (Email)")
        feature_names = vectorizer.get_feature_names_out()
        coefficients = email_model.coef_[0]
        indices = text_vec.nonzero()[1]
        word_scores = [(feature_names[i], coefficients[i]) for i in indices]
        word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:10]
        for word, score in word_scores:
            st.write(f"{'🔴' if score>0 else '🟢'} {word} (score: {score:.2f})")

    # Result thresholds
    st.divider()
    if prob > 0.85:
        st.error(f"⚠️ Phishing ({prob*100:.2f}%)")
    elif prob > 0.6:
        st.warning(f"⚠️ Suspicious ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Legitimate ({(1-prob)*100:.2f}%)")
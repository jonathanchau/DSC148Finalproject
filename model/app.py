import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="News Virality Predictor", page_icon="📰")

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load('news_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'news_model.pkl' not found. Please upload the model file.")
    st.stop()

st.title("📰 News Virality Predictor")
st.markdown("Enter the article's text and metadata. The NLP pipeline will predict if it achieves high social media engagement (60+ shares).")

st.divider()

# Text Inputs
headline = st.text_area("Article Headline", "Obama announces new economic stimulus package")
title = st.text_input("Article Title", "Obama Economy Speech")

# Metadata Inputs
col1, col2 = st.columns(2)
with col1:
    topic = st.selectbox("Topic Category", ['obama', 'economy', 'microsoft', 'palestine', 'other'])
    source = st.selectbox("News Source", ['CNN', 'New York Times', 'Bloomberg', 'Reuters', 'Other'])
    sentiment_headline = st.slider("Headline Sentiment Polarity", -1.0, 1.0, 0.0)

with col2:
    publish_hour = st.slider("Publish Hour (0-23)", 0, 23, 12)
    publish_day = st.selectbox("Publish Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
    sentiment_title = st.slider("Title Sentiment Polarity", -1.0, 1.0, 0.0)

st.divider()

# Prediction Logic
if st.button("Predict Virality", type="primary", use_container_width=True):
    # Calculate derived features required by the pipeline
    headline_len = len(headline.split())
    title_len = len(title.split())
    
    # Construct DataFrame (Must exactly match training features)
    input_data = pd.DataFrame({
        'SentimentTitle': [sentiment_title],
        'SentimentHeadline': [sentiment_headline],
        'TitleLength': [title_len],
        'HeadlineLength': [headline_len],
        'Hour': [publish_hour],
        'DayOfWeek': [publish_day],
        'Topic': [topic],
        'Source_Clean': [source],
        'Title': [title],
        'Headline': [headline]
    })
    
    # Run prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Display Results
    if prediction == 1:
        st.success(f"🔥 **Prediction: POPULAR** (Viral Probability: {probabilities[1]:.1%})")
    else:
        st.error(f"📉 **Prediction: NOT POPULAR** (Viral Probability: {probabilities[1]:.1%})")

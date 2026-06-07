import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="News Virality Predictor", page_icon="📰", layout="centered")

# Load the trained Pipeline safely
@st.cache_resource
def load_pipeline():
    return joblib.load('news_model.pkl')

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("❌ 'news_model.pkl' not found in the root directory.")
    st.stop()

st.title("📰 News Virality Predictor")
st.markdown("Input article attributes below to check if the class-balanced NLP Logistic Regression pipeline predicts high social media engagement.")

st.divider()

# Interactive Inputs
headline = st.text_area("Article Headline", "Obama announces new economic stimulus package to boost employment")
title = st.text_input("Article Title", "Obama Economy Speech")

col1, col2 = st.columns(2)
with col1:
    topic = st.selectbox("Topic Category", ['economy', 'obama', 'microsoft', 'palestine', 'other'])
    source = st.selectbox("News Source", ['Bloomberg', 'Reuters', 'CNN', 'USA TODAY', 'Other'])
    sentiment_headline = st.slider("Headline Sentiment Polarity", -1.0, 1.0, 0.0)

with col2:
    publish_hour = st.slider("Publish Hour (0-23)", 0, 23, 12)
    publish_day = st.selectbox("Publish Day of Week", [0, 1, 2, 3, 4, 5, 6], 
                               format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
    sentiment_title = st.slider("Title Sentiment Polarity", -1.0, 1.0, 0.0)

st.divider()

if st.button("Predict Virality", type="primary", use_container_width=True):
    
    # Construct input dataframe matching the exact 13 columns your model demands
    input_df = pd.DataFrame({
        'Title': [title],
        'Headline': [headline],
        'Source': [source],
        'Topic': [topic],
        'SentimentTitle': [sentiment_title],
        'SentimentHeadline': [sentiment_headline],
        'title_length': [len(title)],                       # Character length
        'headline_length': [len(headline)],                 # Character length
        'title_word_count': [len(title.split())],           # Word count
        'headline_word_count': [len(headline.split())],     # Word count
        'publish_month': [datetime.now().month],            # Defaults to current month
        'publish_dayofweek': [publish_day],
        'publish_hour': [publish_hour]
    })
    
    # Predict using the loaded pipeline
    try:
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
        
        if prediction == 1:
            st.success(f"🔥 **Prediction: POPULAR** (Viral Probability: {probability:.1%})")
        else:
            st.error(f"📉 **Prediction: NOT POPULAR** (Viral Probability: {probability:.1%})")
            
    except Exception as e:
        st.error(f"Prediction failed. Backend error: {e}")

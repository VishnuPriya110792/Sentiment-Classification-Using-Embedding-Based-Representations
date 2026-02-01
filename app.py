import streamlit as st
from sentence_transformers import SentenceTransformer
import joblib

st.title ("Sentiment Analysis App")
tweet = st.text_area("Enter Tweet for Sentiment Analysis")
if st.button("Analyze"):
    # Load the pre-trained model
    model = joblib.load("sentiment_model.pkl")
    # Load the pre-trained SentenceTransformer model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Preprocess the input tweet
    def clean_text(text):
        import re
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove mentions
        text = re.sub(r'#\w+', '', text)     # Remove hashtags
        text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers
        text = str(text).lower()                   # Convert to lowercase
        return text.strip()
    cleaned_tweet = clean_text(tweet)
    # Generate embedding for the input tweet
    tweet_embedding = sbert_model.encode([cleaned_tweet])
    # Predict sentiment
    prediction = model.predict(tweet_embedding)
    st.success(f"The predicted sentiment is: {prediction[0]}")
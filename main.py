import os
import gdown
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st

st.title('🎬 Movie Review Sentiment Analysis')

# Load word index
word_index = imdb.get_word_index()
indexed_word = {v: k for k, v in word_index.items()}

# Model download
file_id = "1UuSftKY5gsw-Azay9dSoUXoIy6pytEyB"
model_path = "model.h5"
download_url = f"https://drive.google.com/uc?id={file_id}"

# Download model if not exists
if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
    with st.spinner('⬇️ Downloading model...'):
        try:
            gdown.download(download_url, model_path, quiet=False)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()
else:
    st.info("✅ Model already downloaded.")

# Load model safely and cache
@st.cache_resource
def load_model_safely():
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

with st.spinner('🔄 Loading model...'):
    model = load_model_safely()
st.success("✅ Model loaded successfully!")

# Text preprocessing
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction
def predict_sentiment(review):
    prediction = model.predict(review, verbose=0)
    sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'
    return sentiment, prediction[0][0]

# UI
user_input = st.text_area('📝 Enter a movie review:')
if st.button('🎯 Classify'):
    if user_input.strip():
        processed_input = preprocess_text(user_input)
        sentiment, score = predict_sentiment(processed_input)
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Rating:** {10 * score:.2f} / 10')
    else:
        st.warning("⚠️ Please enter a review to classify.")
else:
    st.write('👆 Enter a movie review and click **Classify** to see the sentiment.')

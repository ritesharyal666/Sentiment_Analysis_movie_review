import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import os
import gdown
import tensorflow as tf
import streamlit as st

# Google Drive file ID from your link
file_id = "1UuSftKY5gsw-Azay9dSoUXoIy6pytEyB"
model_path = "model.h5"

# Construct direct download URL for gdown
download_url = f"https://drive.google.com/uc?id={file_id}"

# Download the model if not already downloaded
if not os.path.exists(model_path):
    st.write("Downloading model...")
    gdown.download(download_url, model_path, quiet=False)
else:
    st.write("Model already downloaded.")

# Load your model
model = tf.keras.models.load_model(model_path)

word_index = imdb.get_word_index()
indexed_word = {v: k for k, v in word_index.items()}


st.write("Model loaded successfully!")

# Now add your input, preprocessing, prediction code below...

def decoded_review(encoded_review):
  return ' '.join(indexed_word.get(i-3,'?')for i in encoded_review)

def preprocess_text(text):
  words =text.lower().split()
  encoded_review=[word_index.get(word,2)+3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review

def predict_sentiment(review):
  sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
  return sentiment,prediction[0][0]



import streamlit as st
st.title('Movie Review Sentiment Analysis')
st.write('Enter a review')
#user_input
user_input=st.text_area('Movie Review')
if st.button('Classify'):
    preprocess_input= preprocess_text(user_input)
    prediction=model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Rating :{round(prediction[0][0],2)}')
else:
   st.write('Enter a movie review.')
  



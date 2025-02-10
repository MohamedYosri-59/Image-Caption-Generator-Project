import os
os.system("pip install gdown")

# Function to install dependencies from requirements.txt
def install_requirements():
    os.system("pip install -r requirements.txt")

# Install dependencies (optional, but will make sure everything is installed)
install_requirements()


import gdown  # To download files from Google Drive
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array



# Download files from Google Drive without checking if they exist locally
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)



# Google Drive file IDs for the models
vgg16_features_file_id = '1vAhJUjCKhtYVRvmAZTnwovfbigZiUDEm'
caption_model_file_id = '1Eix2lcbdrmow_Andf7LyDdNZyKW1YHew'

# Download the model files from Google Drive
download_file_from_google_drive(vgg16_features_file_id, 'vgg16_features.pkl')
download_file_from_google_drive(caption_model_file_id, 'caption_model.h5')

# Load the saved model and tokenizer
@st.cache_resource
def load_caption_model():
    return tf.keras.models.load_model("caption_model.h5")

@st.cache_data
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_max_length():
    with open("max_length.pkl", "rb") as file:
        return pickle.load(file)

# Load VGG16 model for feature extraction
@st.cache_resource
def load_vgg16():
    model = VGG16(weights="imagenet")
    return tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

# Function to extract features from an image
def extract_features(image, model):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature.flatten().reshape(1, -1)

# Function to generate a caption
def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# Streamlit UI
st.title("Image Caption Generator")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        st.write("Processing...")

        # Load resources
        caption_model = load_caption_model()
        tokenizer = load_tokenizer()
        max_length = load_max_length()
        vgg_model = load_vgg16()

        # Extract image features
        feature = extract_features(image, vgg_model)

        # Generate and display caption
        caption = generate_caption(caption_model, tokenizer, feature, max_length=7)
        st.subheader("Generated Caption:")
        st.write(caption)
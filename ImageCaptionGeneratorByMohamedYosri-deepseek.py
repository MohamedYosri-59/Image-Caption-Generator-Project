import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown
import os

# Download files from Google Drive if they don't exist
if not os.path.exists("vgg16_features.pkl"):
    gdown.download(id="1vAhJUjCKhtYVRvmAZTnwovfbigZiUDEm", output="vgg16_features.pkl", quiet=False)

if not os.path.exists("caption_model.h5"):
    gdown.download(id="1Eix2lcbdrmow_Andf7LyDdNZyKW1YHew", output="caption_model.h5", quiet=False)

# Load the tokenizer and max_length
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("max_length.pkl", "rb") as file:
    max_length = pickle.load(file)

# Load the pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet')
vgg16_model = Model(inputs=vgg16_model.input, outputs=vgg16_model.layers[-2].output)

# Load the caption model
caption_model = load_model("caption_model.h5")

# Function to extract features from an image
def extract_feature_from_image(model, preprocess_fn, image):
    image = image.resize((224, 224))  # Resize image for VGG models
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions
    image = preprocess_fn(image)  # Preprocess for the model
    feature = model.predict(image, verbose=0)  # Extract feature
    return feature.flatten().reshape(1, -1)  # Return as input format

# Function to generate a caption
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'  # Start sequence token
    for _ in range(max_length):  # Generate words up to max_length
        sequence = tokenizer.texts_to_sequences([in_text])[0]  # Convert text to sequence
        sequence = pad_sequences([sequence], maxlen=max_length)  # Pad sequence
        yhat = model.predict([photo, sequence], verbose=0)  # Predict next word
        yhat = np.argmax(yhat)  # Get highest probability word
        word = tokenizer.index_word.get(yhat, None)  # Convert index to word
        if word is None:
            break
        in_text += ' ' + word  # Append word to sequence
        if word == 'endseq':  # Stop if end token is reached
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()  # Return and clean up the output generated caption

# Streamlit app
st.title("Image Caption Generator")
st.write("Upload an image and the model will generate a caption for it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating caption...")

    # Extract features from the image
    feature = extract_feature_from_image(vgg16_model, preprocess_vgg, image)

    # Generate caption
    caption = generate_caption(caption_model, tokenizer, feature, max_length)

    st.write("### Generated Caption:")
    st.write(caption)
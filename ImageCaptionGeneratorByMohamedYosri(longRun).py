import os

#os.system("pip install --upgrade pip")

#Function to install dependencies from requirements.txt
def install_requirements():
    os.system("pip install -r requirements.txt")

# Install dependencies (optional, but will make sure everything is installed)
install_requirements()

#os.system("pip install --user gdown")

import string
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.utils import custom_object_scope
import gdown  # To download files from Google Drive
from collections import OrderedDict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# Download files from Google Drive without checking if they exist locally
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# Google Drive file IDs for the models
vgg16_features_file_id = '1vAhJUjCKhtYVRvmAZTnwovfbigZiUDEm'
#caption_model_file_id = '1Eix2lcbdrmow_Andf7LyDdNZyKW1YHew'

# Download the model files from Google Drive
download_file_from_google_drive(vgg16_features_file_id, 'vgg16_features.pkl')
#download_file_from_google_drive(caption_model_file_id, 'caption_model.h5')

def load_captions(filepath):
    captions = {}  # Dictionary to store captions
    with open(filepath, 'r') as file:  # Open file in read mode
        for line in file:  # Iterate through each line
            line = line.strip()  # Remove leading/trailing spaces
            if not line:  # Skip empty lines
                continue
            image_id, caption = line.split('\t')  # Split line into image ID and caption
            image_id = image_id.split('#')[0]  # Extract image filename
            caption = caption.lower().translate(str.maketrans('', '', string.punctuation))  # Convert caption to lowercase and remove punctuation
            captions.setdefault(image_id, []).append(caption)  # Store captions for each image
    return captions  # Return dictionary of captions
# Load captions
dataset_captions = load_captions("Flickr8k.token.txt")  # Load captions from dataset

@st.cache_data
def load_features():
    with open("vgg16_features.pkl", "rb") as file:
        return pickle.load(file)

features_dict = load_features()

@st.cache_data
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)

tokenizer = load_tokenizer()

vocab_size = len(tokenizer.word_index) + 1  # Define vocabulary size

@st.cache_data
def load_max_length():
    with open("max_length.pkl", "rb") as file:
        return pickle.load(file)

max_length = load_max_length()

# Load VGG16 model for feature extraction
@st.cache_resource
def load_vgg16():
    model = VGG16(weights="imagenet")
    return tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

vgg_model = load_vgg16()

def create_sequences(tokenizer, max_length, dataset_captions, features_dict, vocab_size):
    X1, X2, y = [], [], []

    for img_id, captions in dataset_captions.items():
        if img_id not in features_dict:
            print(f"Skipping {img_id}, feature not found!")  # Debug missing features
            continue

        feature = features_dict[img_id]  # Load precomputed feature
        print(f"Processing {img_id}, Feature Shape: {feature.shape}")  # Debugging step

        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_word = seq[:i], seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]  # Pad sequence
                out_word = to_categorical([out_word], num_classes=vocab_size)[0]  # One-hot encode

                X1.append(feature)  # Image feature
                X2.append(in_seq)   # Text input
                y.append(out_word)  # Next word

    print(f"Total Samples: {len(X1)}")  # Debugging step
    return np.array(X1), np.array(X2), np.array(y)

# Generate Training Data
X1_train, X2_train, y_train = create_sequences(tokenizer, max_length, dataset_captions, features_dict, vocab_size)

# Function to build captioning model
def build_model(vocab_size, max_length):
    inputs = Input(shape=(4096,))  # Input layer for image features
    x = Dense(512, activation='relu')(inputs)  # Dense layer for feature transformation
    x = Dropout(0.2)(x)  # Dropout layer for regularization

    text_input = Input(shape=(max_length,))  # Input layer for text sequences
    embedding = Embedding(vocab_size, 512, mask_zero=True)(text_input)  # Embedding layer
    lstm = LSTM(512, return_sequences=False)(embedding)  # LSTM layer for sequence processing

    decoder = tf.keras.layers.add([x, lstm])  # Merge image and text features
    decoder = Dense(512, activation='relu')(decoder)  # Dense layer
    outputs = Dense(vocab_size, activation='softmax')(decoder)  # Output layer

    model = Model(inputs=[inputs, text_input], outputs=outputs)  # Define model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)  # Compile model
    return model  # Return compiled model

# Build model
caption_model = build_model(vocab_size, max_length)  # Create captioning model
# Train the model
caption_model.fit([X1_train, X2_train], y_train, epochs=1, batch_size=64, verbose=1)

# Function to extract features from an image
def extract_features(image, model):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature.flatten().reshape(1, -1)

# Function to generate a caption

def remove_repeated_words(caption):
    words = caption.split()
    unique_words = list(OrderedDict.fromkeys(words))  # Remove duplicates while maintaining order
    return ' '.join(unique_words)


def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'  # Start sequence token
    repeated_words = set()  # Set to track words used
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
    in_text = remove_repeated_words(in_text)                                                  # Remove repeated words
    return in_text.replace('startseq', '').replace('endseq', '').strip()  # Return and Clean up the output generated caption

# Streamlit UI
st.title("Image Caption Generator")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        st.write("Processing...")

        # Load resources
        caption_model = caption_model
        tokenizer = load_tokenizer()
        max_length = load_max_length()
        vgg_model = load_vgg16()

        # Extract image features
        feature = extract_features(image, vgg_model)

        # Generate and display caption
        caption = generate_caption(caption_model, tokenizer, feature, max_length=5)
        st.subheader("Generated Caption:")
        st.write(caption)
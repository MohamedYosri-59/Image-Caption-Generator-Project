import gdown
import os
import streamlit as st
import pickle
import tensorflow as tf

# Define file IDs from Google Drive (replace these with your actual file IDs)
FILE_IDS = {
    "vgg16_features.pkl": "1vAhJUjCKhtYVRvmAZTnwovfbigZiUDEm",  # Replace with your actual file ID
    "caption_model.h5": "1Eix2lcbdrmow_Andf7LyDdNZyKW1YHew"     # Replace with your actual file ID
}

# Function to download files from Google Drive
def download_file(file_name, file_id):
    if not os.path.exists(file_name):  # Avoid re-downloading
        url = f"https://drive.google.com/uc?id={file_id}"
        st.write(f"Downloading {file_name}...")  # Show download progress in Streamlit
        gdown.download(url, file_name, quiet=False)
        st.success(f"Downloaded {file_name}")

# Download required files
for file, fid in FILE_IDS.items():
    download_file(file, fid)

# Load model
st.write("Loading caption model...")
caption_model = tf.keras.models.load_model("caption_model.h5")
st.success("Caption model loaded!")

# Load precomputed VGG16 features
st.write("Loading VGG16 features...")
with open("vgg16_features.pkl", "rb") as f:
    vgg16_features = pickle.load(f)
st.success("VGG16 features loaded!")

st.write("App is ready! Upload an image to generate captions.")

import streamlit as st
from keras.models import load_model as model_arc  # Ensure this function is defined
import numpy as np
import os
from PIL import Image

# Define the path to your model weights
model_weights_path = './weights/modelnew.h5'

@st.cache_resource(allow_output_mutation=True)  # Use cache_resource for loading the model
def load_model():
    model = model_arc()  # Ensure this function returns the correct model architecture
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)  # Load the pre-trained weights
    else:
        st.error("Model weights file not found. Please check the path.")
    return model

# Load the model once and reuse it
model = load_model()

# Image uploading logic
st.title("Waste Classification Model")
st.write("Upload an image of waste for classification.")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Load and preprocess the image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for your model (you need to adjust this based on your model's input requirements)
    image = image.resize((224, 224))  # Example size; change it as needed
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)

    # Assuming your model outputs class probabilities
    predicted_class = np.argmax(prediction, axis=1)
    st.write(f"Predicted Class: {predicted_class[0]}")  # Adjust based on your class mapping

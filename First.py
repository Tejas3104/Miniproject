import streamlit as st
import numpy as np
import os
from PIL import Image
from utils import preprocess, model_arc, gen_labels

model_weights_path = './weights/modelnew.h5'

@st.cache_resource
def load_model():
    model = model_arc()
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
    else:
        st.error("Model weights file not found. Please check the path.")
    return model

model = load_model()

background_image_url = "https://png.pngtree.com/thumb_back/fh260/background/20220217/pngtree-green-simple-atmospheric-waste-classification-illustration-background-image_953325.jpg"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Waste Classification Model")
st.write("Upload an image of waste for classification.")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader_1")

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    image_array = preprocess(image)

    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction, axis=1)
    
    labels = gen_labels()
    predicted_label = labels[predicted_class[0]]

    st.write(f"Predicted Class: {predicted_label}")

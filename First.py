import streamlit as st
import numpy as np
import os
from PIL import Image
from utils import preprocess, model_arc, gen_labels  # Importing from utils.py

# Define the path to your model weights
model_weights_path = './weights/modelnew.h5'

@st.cache_resource  # Use cache_resource for loading the model
def load_model():
    model = model_arc()  # Ensure this function returns the correct model architecture
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)  # Load the pre-trained weights
    else:
        st.error("Model weights file not found. Please check the path.")
    return model

# Load the model once and reuse it
model = load_model()
import streamlit as st
import numpy as np
import os
from PIL import Image
from utils import preprocess, model_arc, gen_labels  # Importing from utils.py

# Define the path to your model weights
model_weights_path = './weights/modelnew.h5'

@st.cache_resource  # Use cache_resource for loading the model
def load_model():
    model = model_arc()  # Ensure this function returns the correct model architecture
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)  # Load the pre-trained weights
    else:
        st.error("Model weights file not found. Please check the path.")
    return model

# Load the model once and reuse it
model = load_model()

# Set the background image
background_image_url = "https://png.pngtree.com/thumb_back/fh260/background/20220217/pngtree-green-simple-atmospheric-waste-classification-illustration-background-image_953325.jpg"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        color: white; /* Change text color for better visibility */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.title("Waste Classification Model")
st.write("Upload an image of waste for classification.")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Load and preprocess the image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    image_array = preprocess(image)

    # Make prediction
    prediction = model.predict(image_array)

    # Assuming your model outputs class probabilities
    predicted_class = np.argmax(prediction, axis=1)
    
    # Get class labels
    labels = gen_labels()
    predicted_label = labels[predicted_class[0]]

    st.write(f"Predicted Class: {predicted_label}")  # Display the predicted class label

# Set the background image to a waste-related image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.google.com/search?q=waste+image+background+for+my+website&sca_esv=95f18603daa625a6&udm=2&biw=1396&bih=705&sxsrf=ADLYWIK_rAxtMbHifCB9EoqXC8Jyh2XYoA%3A1727896726065&ei=lpz9ZovRA9K_vr0P3Y788AE&ved=0ahUKEwiL9ML0tPCIAxXSn68BHV0HHx4Q4dUDCBE&uact=5&oq=waste+image+background+for+my+website&gs_lp=Egxnd3Mtd2l6LXNlcnAiJXdhc3RlIGltYWdlIGJhY2tncm91bmQgZm9yIG15IHdlYnNpdGVI3TpQlARYzjhwAXgAkAEAmAHVAaABlByqAQYwLjI1LjG4AQPIAQD4AQGYAg2gAskNwgIFEAAYgATCAggQABgHGAoYHsICBhAAGAcYHsICBhAAGAoYHsICBBAAGB7CAggQABgKGB4YD8ICCBAAGAUYChgewgIGEAAYBRgewgIGEAAYCBgewgIIEAAYCBgKGB7CAgoQABgIGAoYHhgPmAMAiAYBkgcGMS4xMS4xoAe1Jw&sclient=gws-wiz-serp#vhid=1O9STm0RXsK0cM&vssid=mosaic")
        background-size: cover;
        background-position: center;
        color: white; /* Change text color for better visibility */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.title("Waste Classification Model")
st.write("Upload an image of waste for classification.")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Load and preprocess the image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    image_array = preprocess(image)

    # Make prediction
    prediction = model.predict(image_array)

    # Assuming your model outputs class probabilities
    predicted_class = np.argmax(prediction, axis=1)
    
    # Get class labels
    labels = gen_labels()
    predicted_label = labels[predicted_class[0]]

    st.write(f"Predicted Class: {predicted_label}")  # Display the predicted class label

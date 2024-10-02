import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *  # Ensure this file contains preprocess(), model_arc(), gen_labels()

# Generate the labels
labels = gen_labels()

# HTML for the title and header
html_temp = '''
  <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -50px">
    <div style = "display: flex; flex-direction: row; align-items: center; justify-content: center;">
     <center><h1 style="color: #000; font-size: 50px;"><span style="color: #0e7d73">Smart </span>Garbage</h1></center>
    <img src="https://cdn-icons-png.flaticon.com/128/1345/1345823.png" style="width: 0px;">
    </div>
    <div style="margin-top: -20px">
    <img src="https://i.postimg.cc/W3Lx45QB/Waste-management-pana.png" style="width: 400px;">
    </div>  
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)

# Subheading for the classifier
html_temp = '''
    <div>
    <center><h3 style="color: #008080; margin-top: -20px">Check the type here</h3></center>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)

# Disable deprecation warning for file uploader encoding
st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload options for the user
opt = st.selectbox("How do you want to upload the image for classification?", 
                   ('Please Select', 'Upload image via link', 'Upload image from device'))

# Function to load and cache the model (so it's loaded only once)
@st.cache(allow_output_mutation=True)
def load_model():
    model = model_arc()  # Ensure this function returns the correct model architecture
    model.load_weights("./weights/modelnew.h5")  # Load the pre-trained model weights
    return model

# Load the model once and reuse it
model = load_model()

# Image uploading logic
image = None
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':
    try:
        img_url = st.text_input('Enter the Image Address')
        if img_url:
            image = Image.open(urllib.request.urlopen(img_url))
    except Exception as e:
        if st.button('Submit'):
            st.error("Please Enter a valid Image Address!")
            time.sleep(4)

# Display the uploaded image
if image is not None:
    st.image(image, width=300, caption='Uploaded Image')

    # Predict button logic
    if st.button('Predict'):
        # Preprocess the image using the utility function
        img = preprocess(image)

        # Model prediction
        prediction = model.predict(img[np.newaxis, ...])

        # Display the result with the class label
        st.info(f'Hey! The uploaded image has been classified as "{labels[np.argmax(prediction[0], axis=-1)]} product".')

# Handle any exceptions gracefully
try:
    # This is moved inside the try block
    pass  # You can handle specific code here if needed
except Exception as e:
    st.info(f"Error: {e}")

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess(image):
    """
    Preprocess the input image for model prediction.
    - Convert RGBA to RGB if needed.
    - Resize to the expected input size of the model.
    - Convert the image to a numpy array.
    - Normalize pixel values (if required).
    """
    # Convert 4-channel RGBA images to 3-channel RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define the target size (modify based on your model's input requirements)
    target_size = (224, 224)  # Example for VGG16
    image = image.resize(target_size)
    
    # Convert image to array
    image = img_to_array(image)
    
    # Add batch dimension and normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    return image

def model_arc():
    """
    Build and return the model architecture.
    Modify this based on the architecture you're using.
    """
    model = Sequential()
    # Example architecture - modify layers as needed
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # Assuming 7 classes for classification
    return model

def gen_labels():
    """
    Generate labels for the classes.
    Modify this list according to your specific classes.
    """
    return ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash", "Compost"]

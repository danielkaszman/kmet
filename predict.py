import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import cv2

# Load the pre-trained model
cwd = Path.cwd()
model_dir = Path(os.path.join(cwd, "model.h5"))
model = tf.keras.models.load_model(model_dir)

# Function to predict the number from the image
def predict_number(image_path, model):
    img = cv2.imread(image_path)
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    predicted_number = np.argmax(prediction)
    
    return predicted_number

# Example usage:
image_path = Path(os.path.join(cwd, "Numbers/4/Four_full (18).jpg")) 
image_path = str(image_path)
predicted_number = predict_number(image_path, model)
print("Predicted number:", predicted_number)

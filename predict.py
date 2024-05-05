import numpy as np
import tensorflow as tf
from pathlib import Path
import keras
import os
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_model(model_dir: str) -> tf.keras.Model:
    """Load a pre-trained model"""
    try:
        model = keras.saving.load_model(model_dir)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_number(image_path: str, model: tf.keras.Model) -> int:
    """Predict the number from an image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")
        
        # Preprocess the image
        img = np.invert(np.array([img]))
        
        # Check model input shape
        input_shape = model.input_shape
        if input_shape[1:]!= img.shape[1:]:
            raise ValueError("Model input shape mismatch")
        
        # Make prediction
        prediction = model.predict(img)
        
        # Function to format a value as percentage with two decimals
        def format_as_percentage(value):
            return f"{value * 100:.2f}%"
        
        percentages = np.vectorize(format_as_percentage)(prediction)
        
        print("Predictions:")
        for i, p in enumerate(percentages[0]):
            print(f"{i}: {p}")
        
        # print(percentages)
        output_shape = model.output_shape
        if output_shape[1:]!= prediction.shape[1:]:
            raise ValueError("Model output shape mismatch")
        
        predicted_number = np.argmax(prediction)
        return predicted_number
    except Exception as e:
        print(f"Error predicting number: {e}")
        return -1

# Example usage:
cwd = Path.cwd()
model_dir = Path(os.path.join(cwd, "model2.keras"))
model = load_model(model_dir)
if model is None:
    print("Failed to load model")
    exit()

image_path = Path(os.path.join(cwd, "Numbers/0/Zero_full (18).jpg")) 
image_path = str(image_path)
predicted_number = predict_number(image_path, model)
print("Predicted number:", predicted_number)
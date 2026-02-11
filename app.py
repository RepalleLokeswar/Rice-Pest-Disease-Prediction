import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image
import io
import base64
app = Flask(__name__)

     # Set paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = r"C:\Users\chand\Music\DP project\dpproject\Rice_leaf_disease_detection.h5"
csv_path = r"C:\Users\chand\Music\DP project\dpproject\description.csv"
json_path = r"C:\Users\chand\Music\DP project\dpproject\class_indices.json"

     # Validate file paths
if not os.path.exists(model_path):
     raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Class indices file not found at {json_path}")
if not os.path.exists(csv_path):
     raise FileNotFoundError(f"Description CSV file not found at {csv_path}")

     # Load model
model = tf.keras.models.load_model(model_path)

     # Load class indices
with open(json_path) as f:
    class_indices = json.load(f)
    class_names = {int(k): v for k, v in class_indices.items()}  # Convert keys to int

     # Load and validate CSV
description_df = pd.read_csv(csv_path)
required_columns = ['disease', 'Description', 'Precautions']
if not all(col in description_df.columns for col in required_columns):
    raise ValueError("description.csv missing required columns: 'disease', 'Description', 'Precautions'")

     # Function to load and preprocess image
def load_and_preprocess_image(image, target_size=(256, 256)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

     # Function to predict class
def predict_image_class(model, image, class_names):
         preprocessed_img = load_and_preprocess_image(image)
         predictions = model.predict(preprocessed_img, verbose=0)
         predicted_class_index = np.argmax(predictions, axis=1)[0]
         predicted_class_name = class_names.get(predicted_class_index, "Unknown")
         confidence = np.max(predictions) * 100
         return predicted_class_name, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
         prediction = None
         confidence = None
         description = None
         precautions = None
         image_data = None
         error = None

         if request.method == 'POST':
             if 'file' not in request.files:
                 error = "No file uploaded"
             else:
                 file = request.files['file']
                 if file.filename == '':
                     error = "No file selected"
                 else:
                     try:
                         image = Image.open(file)
                         predicted_class, confidence = predict_image_class(model, image, class_names)
                         
                         # Get description and precautions from CSV
                         disease_details = description_df[description_df['disease'] == predicted_class]
                         if not disease_details.empty:
                             description = disease_details.iloc[0]['Description']
                             precautions = disease_details.iloc[0]['Precautions'].split(';')
                         else:
                             description = "No additional information available for this disease."
                             precautions = ["Consult an agricultural expert."]
                         
                         # Convert image to base64 for display
                         buffered = io.BytesIO()
                         image.save(buffered, format="PNG")
                         image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

                         prediction = predicted_class
                     except Exception as e:
                         error = f"Error processing image: {str(e)}"
         
         return render_template('index.html', prediction=prediction, confidence=confidence,
                              description=description, precautions=precautions, image_data=image_data, error=error)

if __name__ == '__main__':
         app.run(debug=True)
from keras.models import load_model
from keras.utils import load_img, img_to_array
from flask import Flask, request, render_template
import numpy as np

# Load the model using the correct path
model_path = r"C:\Users\aerwa\OneDrive\Desktop\ML project\tomato_leaf_disease_model.keras"
model = load_model(model_path)

# Class labels for prediction
class_labels = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400
    
    # Preprocess the uploaded image
    img = load_img(file, target_size=(224, 224))  # Resize the image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    return f"Predicted Class: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)

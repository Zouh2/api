from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load your model
model = tf.lite.Interpreter(model_path='compressed_model.h5')
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0).astype(np.float32)

def predict(image):
    labels = ['Healthy', 'Powdery', 'Rust']
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    preds = model.get_tensor(output_details[0]['index'])
    preds_class = np.argmax(preds)
    preds_label = labels[preds_class]
    confidence_score = float(preds[0][preds_class])
    return preds_label, confidence_score

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            img = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(img)
            label, confidence = predict(processed_image)
            return jsonify({'predicted_class': label, 'confidence_score': confidence})
    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

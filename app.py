from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the pre-trained model
model = load_model('digits_recognition_cnn.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', error='No image file selected')

    # Read and preprocess the image
    try:
        image = Image.open(image_file)
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image) / 255.0  # Normalize pixel values
    except Exception as e:
        return render_template('index.html', error=str(e))

    # Reshape the image array to match the model's input shape
    image_array = image_array.reshape((1, 28, 28, 1))  # Add batch dimension and reshape
    
    # Make prediction
    prediction = model.predict(image_array)
    predicted_number = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)  # Confidence in percentage

    # Encode the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Render the result directly in the index template
    return render_template('index.html', image=img_str, predicted_number=predicted_number, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)

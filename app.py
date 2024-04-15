from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pre-trained model
model = load_model('digits_recognition_cnn.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.files['image'].read()
    
    # Resize the image to 28x28 pixels
    try:
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((28, 28))
    except Exception as e:
        return jsonify({'error': 'Error resizing image: {}'.format(str(e))})
    
    # Convert the image to grayscale
    img_gray = img.convert('L')
    
    # Convert the image to a numpy array
    image_array = np.array(img_gray)
    
    # Check if the image size is not equal to 28*28
    if image_array.shape != (28, 28):
        return jsonify({'error': 'Image must be 28x28 pixels after resizing'})
    
    # Reshape the image array to match the model's input shape
    image_array = image_array.reshape((1, 28, 28, 1))  # Add batch dimension and reshape
    
    # Make prediction
    prediction = model.predict(image_array)
    
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)
    
    # Plot the image
    plt.imshow(img_gray, cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.axis('off')
    plt.show()
    
    # Return the predicted digit
    return jsonify({'prediction': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)

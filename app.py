# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    features = [float(x) for x in request.form.values()]
    # Convert input data into a numpy array
    input_data = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    return render_template('index.html', prediction_text=f'Predicted class: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)

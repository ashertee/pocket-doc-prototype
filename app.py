from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# symptoms to indices
symptom_to_index = {
    "fever": 0,
    "cough": 10,
    "fatigue": 12,
    "headache": 6,
    'rash': 8,
    'dizziness': 7,
    'blackouts': 3,
    'pain': 9,
    'breathlessness': 4,
    'soreness': 1,

}

# Mock function to preprocess symptoms
def preprocess_symptoms(symptoms):
    symptom_vector = np.zeros(100)  # Assuming 100 possible symptoms
    for symptom in symptoms:
        index = symptom_to_index.get(symptom.lower())  # Convert symptom to an index
        if index is not None:
            symptom_vector[index] = 1
    return np.array([symptom_vector])

# Mock function for mapping predictions to diagnoses
def map_prediction_to_diagnosis(prediction):
    diagnoses = ["Flu", "Common Cold", "COVID-19", "Allergy"]
    return diagnoses[np.argmax(prediction)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    input_data = preprocess_symptoms(symptoms)
    prediction = model.predict(input_data)
    diagnosis = map_prediction_to_diagnosis(prediction)
    return jsonify({'diagnosis': diagnosis})

if __name__ == '__main__':
    app.run(debug=True)

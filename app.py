from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('heart_disease_model.pkl')
except FileNotFoundError:
    print("Error: Model file 'heart_disease_model.pkl' not found.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check if the model file exists.'}), 500

    try:
        # Capture the data
        data = {
            'age': int(request.form['age']),
            'sex': request.form['sex'],
            'cp': request.form['cp'],
            'trestbps': int(request.form['trestbps']),
            'chol': int(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': int(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }

        # Create DataFrame
        df = pd.DataFrame({
            'age': [data['age']],
            'trestbps': [data['trestbps']],
            'chol': [data['chol']],
            'fbs': [data['fbs']],
            'restecg': [data['restecg']],
            'thalach': [data['thalach']],
            'exang': [data['exang']],
            'oldpeak': [data['oldpeak']],
            'slope': [data['slope']],
            'ca': [data['ca']],
            'thal': [data['thal']],
            'sex_male': [1 if data['sex'] == 'male' else 0],
            'cp_atypical angina': [1 if data['cp'] == 'atypical angina' else 0],
            'cp_non-anginal pain': [1 if data['cp'] == 'non-anginal pain' else 0],
            'cp_typical angina': [1 if data['cp'] == 'typical angina' else 0]
        })

        # Prediction
        prediction = model.predict(df)
        result = "No Heart Disease" if prediction[0] == 1 else "Heart Disease"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

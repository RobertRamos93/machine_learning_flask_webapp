# Dependencias
from flask import Flask, request, render_template
import joblib
import pandas as pd

# Crear instancia de Flask
app = Flask(__name__)

# Carga del modelo
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Captura de los datos
    age = int(request.form['age'])
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    # DataFrame con las características (el orden importa, revisar el archivo csv después del entrenamiento)
    data = pd.DataFrame({
        'age': [age],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal],
        'sex_male': [1 if sex == 'male' else 0],
        'cp_atypical angina': [1 if cp == 'atypical angina' else 0],
        'cp_non-anginal pain': [1 if cp == 'non-anginal pain' else 0],
        'cp_typical angina': [1 if cp == 'typical angina' else 0]
    })

    # Predicción
    prediction = model.predict(data)
    result = "No Heart disease" if prediction[0] == 1 else "Heart disease"
    return {'result': result}  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

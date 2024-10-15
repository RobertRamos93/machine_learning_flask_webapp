# Dependencias
from flask import Flask, request
from flask_restful import Resource, Api
import joblib
import pandas as pd

# Crear instancia de Flask y Api
app = Flask(__name__)
api = Api(app)

# Carga del modelo
model = joblib.load('heart_disease_model.pkl')

class Predict(Resource):
    def post(self):
        # Captura de los datos
        data = request.get_json()

        # Crear DataFrame con las características
        df = pd.DataFrame({
            'age': [int(data['age'])],
            'trestbps': [int(data['trestbps'])],
            'chol': [int(data['chol'])],
            'fbs': [int(data['fbs'])],
            'restecg': [int(data['restecg'])],
            'thalach': [int(data['thalach'])],
            'exang': [int(data['exang'])],
            'oldpeak': [float(data['oldpeak'])],
            'slope': [int(data['slope'])],
            'ca': [int(data['ca'])],
            'thal': [int(data['thal'])],
            'sex_male': [1 if data['sex'] == 'male' else 0],
            'cp_atypical angina': [1 if data['cp'] == 'atypical angina' else 0],
            'cp_non-anginal pain': [1 if data['cp'] == 'non-anginal pain' else 0],
            'cp_typical angina': [1 if data['cp'] == 'typical angina' else 0]
        })

        # Predicción
        prediction = model.predict(df)
        result = "No Heart disease" if prediction[0] == 1 else "Heart disease"
        return {'result': result}

# Agregar la ruta de la API
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import joblib
import pandas as pd

app = Flask(__name__)
api = Api(app)

# Carga del modelo
model = joblib.load('heart_disease_model.pkl')

class Predict(Resource):
    def post(self):
        # Captura de los datos
        data = request.get_json()
        
        age = int(data['age'])
        sex = data['sex']
        cp = data['cp']
        trestbps = int(data['trestbps'])
        chol = int(data['chol'])
        fbs = int(data['fbs'])
        restecg = int(data['restecg'])
        thalach = int(data['thalach'])
        exang = int(data['exang'])
        oldpeak = float(data['oldpeak'])
        slope = int(data['slope'])
        ca = int(data['ca'])
        thal = int(data['thal'])

        # DataFrame con las características
        df = pd.DataFrame({
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
        prediction = model.predict(df)
        result = "No Heart disease" if prediction[0] == 1 else "Heart disease"
        return jsonify({'result': result})

# Configuración de las rutas
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)

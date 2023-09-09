from flask import Flask, render_template, request
import numpy as np
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_clasificativo.pkl')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Nitro = int(request.form['Nitrógeno'])
    Foro = int(request.form['Fósforo'])
    Pota = int(request.form['Potasio'])
    Tempe = int(request.form['Temperatura'])
    Hume = int(request.form['Humedad'])
    PH = int(request.form['PH_Suelo'])
    Predic = int(request.form['Precipitación'])

    new_samples = np.array([[Nitro, Foro, Pota, Tempe, Hume, PH, Predic]])

    prediction = model.predict(new_samples)

    mensaje = "La clasificación de la etiqueta es: "
    mensaje += prediction[0]

    return render_template('result.html', predi=mensaje)

if __name__ == '__main__':
    app.run()
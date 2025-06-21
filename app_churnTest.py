## app for customer churn

from flask import Flask, request, jsonify, render_template_string
#import pickle
import numpy as np
import joblib

app = Flask(__name__)  # Inicializa la aplicación Flask

model = joblib.load('model/model_churn.pkl') ## made in train_model_churn.py

# HTML form for browser interaction
HTML_FORM = '''
    <h2>Predict Customer Churn</h2>
    <h3> Today: 20 June </h3>
    <form method="POST" action="/predict">
         
        <label>Meses de permanencia (tenure):</label><br>
        <input type="number" name="tenure" required><br><br>

        <label>Cargos mensuales (monthly charges):</label><br>
        <input type="number" step="any" name="monthly_charges" required><br><br>

        <label>Tipo de contrato:</label><br>
        <select name="contract">
            <option value="0">Mes a mes</option>
            <option value="1">Contrato anual o bianual</option>
        </select><br><br>

      <input type="submit">
    </form>
'''
# Formulario (GET)
@app.get("/")
def formulario():
    return HTML_FORM

@app.get("/predict")
def formulario2():
    return HTML_FORM

@app.post("/predict")

def prediccion():
    prediction = None
    probability = None
    
    ## datos del formulario
    tenure= int(request.form['tenure'])
    monthly_charges= float(request.form['monthly_charges'])
    contract=int(request.form['contract'])
        
    # preparar entrada (input) para el modelo
    features = np.array([[tenure,monthly_charges,contract]])

    # Predicción
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    prediction = 'Sí' if pred==1 else 'No'
    probability=round(prob*100,2)

    return render_template_string("""
        <h2>Resultado:</h2>
        <p>¿Abandona el cliente?: <strong>{{ prediccion }}</strong></p>
        <p>Probabilidad de abandono: <strong>{{ probabilidad }}%</strong></p>
         <a href='/'>Volver</a>  """, prediccion= prediction, probabilidad = probability)

# Ejecutar app
if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
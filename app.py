from flask import Flask, render_template, request, redirect, url_for, flash
import os
import torch
import joblib
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from model import GastosMLP

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "../templates"))
app.config['UPLOAD_FOLDER'] = '../uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = 'supersecretkey'

# Cargar modelo y scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/modelo_gastos.pth")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")
try:
    scaler = joblib.load(SCALER_PATH)
    model = GastosMLP(input_size=6)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")

def predecir(datos_entrada):
    datos_entrada = np.array(datos_entrada).reshape(1, -1)
    datos_entrada = scaler.transform(datos_entrada)
    datos_tensor = torch.tensor(datos_entrada, dtype=torch.float32)
    with torch.no_grad():
        prediccion = model(datos_tensor).numpy()
    return round(prediccion[0][0], 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        edad = int(request.form['Edad'])
        salario = float(request.form['Salario'])
        dependientes = int(request.form['Dependientes'])
        educacion = int(request.form['Educacion'])
        vivienda = int(request.form['Vivienda'])
        region = int(request.form['Region'])

        # Suponiendo que los gastos fijos y variables son un % del salario
        gastos_fijos = salario * 0.4  # 40% como ejemplo
        gastos_variables = salario * 0.3  # 30% como ejemplo
        ingreso_disponible = salario - gastos_fijos
        ratio_endeudamiento = (gastos_fijos / salario) * 100
        ahorro_mensual = ingreso_disponible - gastos_variables

        # Simulación de datos de gastos mensuales
        labels = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
        predicciones = [round(predecir([edad, salario, dependientes, educacion, vivienda, region]) * (0.8 + 0.4 * i/11), 2) for i in range(12)]

        indicadores = {
            "ingreso_disponible": round(ingreso_disponible, 2),
            "ratio_endeudamiento": round(ratio_endeudamiento, 2),
            "ahorro_mensual": round(ahorro_mensual, 2),
        }

        return render_template('analysis.html',
                               prediction=f"$ {np.mean(predicciones):.2f}",
                               labels=labels,
                               predicciones=predicciones,
                               indicadores=indicadores)
    except Exception as e:
        flash(f"Error en la predicción: {str(e)}")
        return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        flash('Nombre de archivo vacío o formato incorrecto')
        return redirect(url_for('index'))

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        data = pd.read_csv(filepath)
        if data.shape[1] < 6:
            flash('El archivo CSV no contiene todas las columnas necesarias')
            return redirect(url_for('index'))

        predicciones = [float(predecir(row[:6])) for row in data.itertuples(index=False)]
        labels = [f"Fila {i+1}" for i in range(len(predicciones))]
        
        print("Redirigiendo a analysis.html con datos:", labels, predicciones)

        return render_template('analysis.html',
                               prediction=f"$ {np.mean(predicciones):.2f}",
                               labels=labels,
                               predicciones=predicciones)
    except Exception as e:
        flash(f"Error al procesar el archivo: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
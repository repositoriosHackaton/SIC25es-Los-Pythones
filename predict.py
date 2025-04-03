import os
import torch
import joblib
import numpy as np
from model import GastosMLP

# Obtener la ruta absoluta del directorio actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Definir rutas absolutas
MODEL_PATH = os.path.join(BASE_DIR, "../models/modelo_gastos.pth")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

# Cargar el scaler
scaler = joblib.load(SCALER_PATH)

# Definir y cargar el modelo
model = GastosMLP(input_size=6)  # Ajustar al número correcto de features
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def predecir(datos_entrada):
    datos_entrada = np.array(datos_entrada).reshape(1, -1)
    datos_entrada = scaler.transform(datos_entrada)  # Normalización
    datos_tensor = torch.tensor(datos_entrada, dtype=torch.float32)
    prediccion = model(datos_tensor).detach().numpy()
    return round(prediccion[0][0], 2)

# Prueba de predicción
test_input = [[18, 150, 1, 3, 2, 1]]  # Datos de prueba
print(f"Predicción de gasto: ${predecir(test_input):.2f}")
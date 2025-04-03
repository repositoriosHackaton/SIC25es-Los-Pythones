import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import StandardScaler
from model import GastosMLP

# Cargar datos
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/dataset_gastos_mensuales.csv")
df = pd.read_csv(DATA_PATH)

# Separar características y etiqueta
X = df.iloc[:, :-1].values  # Todas las columnas menos la última (entrada)
y = df.iloc[:, -1].values   # Última columna (salida)

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar el escalador
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/scaler.pkl")
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler guardado en {SCALER_PATH}")

# Definir modelo
input_size = X.shape[1]
model = GastosMLP(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# Guardar el modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/modelo_gastos.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"Modelo guardado en {MODEL_PATH}")


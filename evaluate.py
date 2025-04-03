import torch
import torch.nn as nn
from data_loader import cargar_datos
from model import GastosMLP
import os

MODEL_PATH = os.path.join("models", "modelo_gastos.pth")

def evaluar_modelo():
    (_, X_test, _, y_test), _, _ = cargar_datos()
    
    model = GastosMLP(X_test.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    criterion = nn.MSELoss()
    with torch.no_grad():
        y_pred = model(X_test)
        loss = criterion(y_pred, y_test)
    
    print("Error cuadrático medio:", loss.item())

if __name__ == "__main__":
    evaluar_modelo()

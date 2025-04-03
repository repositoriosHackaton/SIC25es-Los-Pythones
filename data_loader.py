import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("data", "dataset_gastos_mensuales.csv")

def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Gastos_Mensuales'])
    y = df['Gastos_Mensuales'].values.reshape(-1, 1)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    return train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42), scaler_X, scaler_y
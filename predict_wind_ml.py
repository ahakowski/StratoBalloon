import torch
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

from model_definition import WindLSTM  # Zawiera klasę modelu

# Folder gdzie są zapisane modele i skalery
MODEL_DIR = "models"
WINDOW_SIZE = 7

# Mapa ciśnień do wysokości
pressure_to_height = {
    1000: 0.1, 925: 0.7, 850: 1.5, 700: 3.0, 500: 5.5, 400: 7.2, 300: 9.5,
    250: 10.8, 200: 12.0, 150: 13.5, 100: 16.0, 70: 18.5, 50: 20.5,
    30: 23.0, 20: 26.0, 10: 30.0
}

def closest_pressure_for_altitude(altitude_m):
    height_km = altitude_m / 1000
    return min(pressure_to_height.keys(), key=lambda p: abs(pressure_to_height[p] - height_km))

# Główna funkcja predykcyjna
def predict_wind_ml(input_df, pressure_hpa=None, altitude_m=None):
    assert pressure_hpa or altitude_m, "Musisz podać pressure_hpa lub altitude_m"

    if altitude_m is not None:
        pressure_hpa = closest_pressure_for_altitude(altitude_m)

    model_path = os.path.join(MODEL_DIR, f"model_{pressure_hpa}hPa.pt")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{pressure_hpa}hPa.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Brak {model_path} modelu lub skalera dla poziomu {pressure_hpa} hPa")
    
    # Wczytaj model i scaler
    model = WindLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler = joblib.load(scaler_path)

    # Przygotuj dane wejściowe (ostatnie 7 dni)
    input_df = input_df.sort_values("Date")
    input_df = input_df.tail(WINDOW_SIZE)
    input_data = input_df[["WindSpeed_m_s", "WindDir_deg"]].values
    input_scaled = scaler.transform(input_data)
    X_input = torch.tensor(input_scaled[np.newaxis, :, :], dtype=torch.float32)

    with torch.no_grad():
        pred_speed_scaled = model(X_input).numpy()
        pred_speed_unscaled = scaler.inverse_transform(
            np.hstack([pred_speed_scaled, np.zeros_like(pred_speed_scaled)])
        )[:, 0][0]

    return {
        "Pressure_hPa": pressure_hpa,
        "Predicted_WindSpeed_m_s": round(pred_speed_unscaled, 2)
    }

# Przykład użycia (po załadowaniu input_df z 7 ostatnich dni dla danego poziomu):
# result = predict_wind_ml(input_df, altitude_m=10000)
# print(result)

# Punkt 2: Załaduj modele i skalery z dysku na starcie aplikacji
import torch
from torch import nn
import joblib
import os

# Zakładamy, że pliki zapisane są jako:
#   - models/model_{pressure}hPa.pth
#   - models/scaler_{pressure}hPa.pkl

MODEL_DIR = "models"
levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]

class WindLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super(WindLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

models_by_pressure = {}
scalers_by_pressure = {}

for pressure in levels:
    model_path = os.path.join(MODEL_DIR, f"model_{pressure}hPa.pt")  # <- rozszerzenie .pt
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{pressure}hPa.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = WindLSTM()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        scaler = joblib.load(scaler_path)

        models_by_pressure[pressure] = model
        scalers_by_pressure[pressure] = scaler
    else:
        print(f"⚠️ Brak pliku modelu {model_path} lub skalera dla poziomu {pressure} hPa")

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib  # <- do zapisu skalera

# Mapowanie ciśnienia na wysokość
pressure_to_height = {
    1000: 0.1, 925: 0.7, 850: 1.5, 700: 3.0, 500: 5.5, 400: 7.2, 300: 9.5,
    250: 10.8, 200: 12.0, 150: 13.5, 100: 16.0, 70: 18.5, 50: 20.5,
    30: 23.0, 20: 26.0, 10: 30.0
}
levels = list(pressure_to_height.keys())
levels.sort(reverse=True)

# Foldery na modele, wykresy
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

csv_path = "gfs_data.csv"
df = pd.read_csv(csv_path)

target_lat = 52.4
target_lon = 16.9
results_summary = []

class WindLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super(WindLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

for pressure in levels:
    df_filtered = df[(df["Pressure_hPa"] == pressure)].copy()
    df_filtered["lat_diff"] = np.abs(df_filtered["Latitude"] - target_lat)
    df_filtered["lon_diff"] = np.abs(df_filtered["Longitude"] - target_lon)
    df_filtered["total_diff"] = df_filtered["lat_diff"] + df_filtered["lon_diff"]
    df_filtered = df_filtered[df_filtered["total_diff"] == df_filtered["total_diff"].min()].copy()

    df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], format="%Y%m%d")
    df_filtered.sort_values("Date", inplace=True)

    if len(df_filtered) < 10:
        continue  # zbyt mało danych

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_filtered[["WindSpeed_m_s", "WindDir_deg"]])
    scaled_df = pd.DataFrame(scaled, columns=["WindSpeed", "WindDir"])

    def create_sequences(data, window_size=7):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_df.values, window_size=7)
    if len(X) < 10:
        continue

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=8, shuffle=True)

    model = WindLSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).detach().numpy()
        y_test_np = y_test_t.numpy()

    y_pred_unscaled = scaler.inverse_transform(np.hstack([y_pred, np.zeros_like(y_pred)]))[:, 0]
    y_test_unscaled = scaler.inverse_transform(np.hstack([y_test_np, np.zeros_like(y_test_np)]))[:, 0]

    # Zapis wykresu
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_unscaled, label="Rzeczywiste")
    plt.plot(y_pred_unscaled, label="Predykcja")
    plt.title(f"LSTM – Predykcja wiatru – {pressure} hPa ({pressure_to_height[pressure]} km)")
    plt.xlabel("Krok czasowy")
    plt.ylabel("Prędkość [m/s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/prediction_{pressure}hPa.png")
    plt.close()

    # Zapis modelu i skalera
    torch.save(model.state_dict(), f"models/model_{pressure}hPa.pt")
    joblib.dump(scaler, f"models/scaler_{pressure}hPa.pkl")

    results_summary.append({
        "Pressure_hPa": pressure,
        "Height_km": pressure_to_height[pressure],
        "MAE": np.mean(np.abs(y_pred_unscaled - y_test_unscaled)),
        "Samples": len(y_test_unscaled)
    })

# Podsumowanie
summary_df = pd.DataFrame(results_summary).sort_values("Height_km")
print("📊 Podsumowanie wyników modelu LSTM:\n")
print(summary_df.to_string(index=False))

# Eksport do CSV
summary_df.to_csv("results_summary.csv", index=False)
print("\n✅ Wyniki zostały zapisane do pliku: results_summary.csv")

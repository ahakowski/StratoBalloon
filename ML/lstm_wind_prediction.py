
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Ścieżka do pliku CSV z danymi
csv_path = "gfs_data.csv"  # <- Zmień na swoją ścieżkę
df = pd.read_csv(csv_path)

# Filtrowanie danych dla Poznania i poziomu 100 hPa
target_lat = 52.4
target_lon = 16.9
target_pressure = 100
print("Dostępne poziomy ciśnienia:", df["Pressure_hPa"].unique())
print("Latitude unique:", df["Latitude"].unique())
print("Longitude unique:", df["Longitude"].unique())
df_filtered = df[
    (df["Pressure_hPa"] == target_pressure)
].copy()

# Znajdź najbliższe współrzędne
df_filtered["lat_diff"] = np.abs(df_filtered["Latitude"] - target_lat)
df_filtered["lon_diff"] = np.abs(df_filtered["Longitude"] - target_lon)
df_filtered["total_diff"] = df_filtered["lat_diff"] + df_filtered["lon_diff"]

# Wybierz tylko wiersze z najmniejszą różnicą
min_diff = df_filtered["total_diff"].min()
df_filtered = df_filtered[df_filtered["total_diff"] == min_diff].copy()


df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], format="%Y%m%d")
df_filtered.sort_values("Date", inplace=True)

# Skalowanie danych
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_filtered[["WindSpeed_m_s", "WindDir_deg"]])
scaled_df = pd.DataFrame(scaled, columns=["WindSpeed", "WindDir"])

# Tworzenie sekwencji dla LSTM
def create_sequences(data, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])  # WindSpeed
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_df.values, window_size=7)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Tensory
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=8)

# Model LSTM
class WindLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super(WindLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = WindLSTM()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Trenowanie
for epoch in range(30):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Ewaluacja
# Ewaluacja
model.eval()
with torch.no_grad():
    # Przypisujemy wynik do zmiennej 'y_pred'
    y_pred = model(X_test_t).detach().numpy()
    y_test_np = y_test_t.numpy()

# Odwracamy skalowanie - dla uproszczenia przyjmujemy, że pracujemy tylko z jedną cechą (WindSpeed)
# Jeśli używasz scalera na dwie cechy, musisz rozbudować to odpowiednio.
y_pred_unscaled = scaler.inverse_transform(np.hstack([y_pred, np.zeros_like(y_pred)]))[:, 0]
y_test_unscaled = scaler.inverse_transform(np.hstack([y_test_np, np.zeros_like(y_test_np)]))[:, 0]

# Wizualizacja
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(y_test_unscaled, label="Rzeczywiste")
plt.plot(y_pred_unscaled, label="Predykcja")
plt.title("LSTM - Predykcja prędkości wiatru (100 hPa)")
plt.xlabel("Krok czasowy")
plt.ylabel("Prędkość wiatru [m/s]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


with torch.no_grad():
    predictions = model(X_test_t).numpy()
    true = y_test_t.numpy()

# Odwrócenie skalowania
pred_inv = scaler.inverse_transform(np.hstack([predictions, np.zeros_like(predictions)]))[:, 0]
true_inv = scaler.inverse_transform(np.hstack([true, np.zeros_like(true)]))[:, 0]

# Zapis wyników
results_df = pd.DataFrame({
    "True_WindSpeed": true_inv,
    "Predicted_WindSpeed": pred_inv
})
results_df.to_csv("lstm_wind_predictions.csv", index=False)

# Wykres
plt.figure(figsize=(10, 5))
plt.plot(true_inv, label="True", linewidth=2)
plt.plot(pred_inv, label="Predicted", linewidth=2)
plt.title("LSTM - Predykcja prędkości wiatru (100 hPa)")
plt.xlabel("Dzień testowy")
plt.ylabel("Wind speed [m/s]")
plt.legend()
plt.tight_layout()
plt.savefig("lstm_wind_plot.png")
plt.show()

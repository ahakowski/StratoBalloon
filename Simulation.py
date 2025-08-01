# Simulation_ML_Final_Pelna_Struktura.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

# ------------------ MODEL LSTM ------------------
class WindLSTM(torch.nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super(WindLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ------------------ AUTOREGRESJA ------------------
def predict_autoregressive(pressure_hpa: int, days_ahead: int = 1) -> float:
    model_path = f"models/model_{pressure_hpa}hPa.pt"
    scaler_path = f"models/scaler_{pressure_hpa}hPa.pkl"

    scaler = joblib.load(scaler_path)
    model = WindLSTM(input_size=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    seed_sequence = np.array([[5.0, 270.0]] * 7)
    scaled = scaler.transform(seed_sequence)
    sequence = scaled.copy()

    for step in range(days_ahead):
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).squeeze(0).numpy()
        new_dir = (sequence[-1][1] + 2 * (step + 1)) % 360
        sequence = np.vstack([sequence[1:], [pred[0], new_dir]])

    predicted_scaled = sequence[-1]
    predicted_full = scaler.inverse_transform([predicted_scaled])[0]
    return predicted_full[0], predicted_full[1]

def wind_speed_dir_to_uv(speed: float, direction_deg: float) -> tuple:
    dir_rad = np.radians(direction_deg)
    u = -speed * np.sin(dir_rad)
    v = -speed * np.cos(dir_rad)
    return u, v

def pressure_to_altitude(pressure_hpa):
    return round(44330 * (1 - (pressure_hpa / 1013.25)**0.1903))

def interpolate_wind(wind_profile, altitude):
    altitudes = wind_profile["Altitude_m"].values
    speeds = wind_profile["WindSpeed_m_s"].values
    dirs = wind_profile["WindDir_deg"].values
    return np.interp(altitude, altitudes, speeds), np.interp(altitude, altitudes, dirs)

def simulate_balloon(lat, lon, ascent_rate, descent_rate, burst_altitude, wind_profile):
    data = []
    current_lat, current_lon = lat, lon
    t = 0

    for h in range(100, burst_altitude + 100, 100):
        wind_speed, wind_dir = interpolate_wind(wind_profile, h)
        dt = 100 / ascent_rate
        t += dt
        dx = (wind_speed) * dt * np.cos(np.deg2rad(270 - wind_dir)) / 111000
        dy = (wind_speed) * dt * np.sin(np.deg2rad(270 - wind_dir)) / 111000
        current_lat += dy
        current_lon += dx
        burst_flag = "Yes" if h == burst_altitude else ""
        data.append((t / 60, current_lat, current_lon, h, wind_speed, wind_dir, burst_flag))

    for h in range(burst_altitude - 100, 0, -100):
        wind_speed, wind_dir = interpolate_wind(wind_profile, h)
        dt = 100 / abs(descent_rate)
        t += dt
        dx = (wind_speed) * dt * np.cos(np.deg2rad(270 - wind_dir)) / 111000
        dy = (wind_speed) * dt * np.sin(np.deg2rad(270 - wind_dir)) / 111000
        current_lat += dy
        current_lon += dx
        data.append((t / 60, current_lat, current_lon, h, wind_speed, wind_dir, ""))

    return pd.DataFrame(
        data,
        columns=["Time_min", "Latitude", "Longitude", "Altitude_m", "WindSpeed_m_s", "WindDir_deg", "Burst"]
    )

# ------------------ STREAMLIT GUI ------------------
st.set_page_config(page_title="Symulacja balonu Stratosferycznego 🎈")
st.title("Symulacja balonu Stratosferycznego 🎈")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Data startu", value=datetime.utcnow().date() + timedelta(days=1))
    start_time = st.time_input("Godzina startu", value=datetime.utcnow().time())
    start_dt = datetime.combine(start_date, start_time)
    ascent_rate = st.number_input("Prędkość wznoszenia (m/s):", 1.0, 20.0, 5.0)
    descent_rate = st.number_input("Prędkość opadania (m/s):", -30.0, -1.0, -8.0)
    burst_altitude = st.number_input("Wysokość pęknięcia (m):", 1000, 35000, 30000)
with col2:
    lat = st.number_input("Szerokość geograficzna", value=52.0)
    lon = st.number_input("Długość geograficzna", value=16.9)

simulate_button = st.button("🚀 Rozpocznij symulację")

if simulate_button:
    today = datetime.utcnow().date()
    days_ahead = (start_date - today).days
    pressure_levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
    wind_profile_rows = []
    for pressure in pressure_levels:
        try:
            speed, direction = predict_autoregressive(pressure, days_ahead=days_ahead)
            alt = pressure_to_altitude(pressure)
            wind_profile_rows.append({
                "Pressure_hPa": pressure,
                "Altitude_m": alt,
                "WindSpeed_m_s": speed,
                "WindDir_deg": direction
            })
        except Exception as e:
            st.error(f"Błąd dla {pressure} hPa: {e}")

    if len(wind_profile_rows) > 0:
        wind_df = pd.DataFrame(wind_profile_rows).sort_values("Altitude_m")
        trajectory_df = simulate_balloon(lat, lon, ascent_rate, descent_rate, burst_altitude, wind_df)
        st.session_state.wind_profile = wind_df
        st.session_state.trajectory_df = trajectory_df
        st.session_state.sim_time = start_dt.strftime('%Y-%m-%d %H:%M')

# --- ZAWSZE pokazuj dane z sesji, jeśli są dostępne ---
if "trajectory_df" in st.session_state and st.session_state.trajectory_df is not None:
    st.markdown(f"### Wynik symulacji dla {st.session_state.sim_time}")
    st.dataframe(st.session_state.wind_profile)
    st.dataframe(st.session_state.trajectory_df)

    csv = st.session_state.trajectory_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Pobierz trajektorię (CSV)", data=csv, file_name="trajektoria.csv", mime="text/csv")

    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)

    burst_rows = st.session_state.trajectory_df[st.session_state.trajectory_df["Burst"] == "Yes"]
    if not burst_rows.empty:
        burst_row = burst_rows.iloc[0]
        folium.Marker(
            [burst_row["Latitude"], burst_row["Longitude"]],
            tooltip=f"Pęknięcie ({burst_altitude} m)",
            icon=folium.Icon(color="blue")
        ).add_to(m)

    folium.PolyLine(st.session_state.trajectory_df[["Latitude", "Longitude"]].values, color="black").add_to(m)
    folium.Marker([
        st.session_state.trajectory_df.iloc[-1]["Latitude"],
        st.session_state.trajectory_df.iloc[-1]["Longitude"]
    ],
        tooltip="Lądowanie", icon=folium.Icon(color="red")
    ).add_to(m)

    st_folium(m, width=700, height=500, key="balloon_map")
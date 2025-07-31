import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
import tempfile
import requests
import torch
from streamlit_folium import st_folium
import folium
from predict_wind_ml import predict_wind_ml
from model_definition import WindLSTM
from gfs_utils import download_and_parse_gfs_for_window, load_input_for_ml

# --- Konfiguracja strony ---
st.set_page_config(page_title="Symulacja balonu Stratosferycznego 🎈", page_icon="🎈")

st.title("Symulacja balonu Stratosferycznego 🎈")

# --- Wybór parametrów użytkownika ---
with st.sidebar:
    st.header("Parametry symulacji")
    start_lat = st.number_input("Szerokość geograficzna (Start):", -90.0, 90.0, 52.4)
    start_lon = st.number_input("Długość geograficzna (Start):", -180.0, 180.0, 16.9)

    user_date = st.date_input("Data startu:", datetime.utcnow().date())
    user_time = st.time_input("Godzina startu (UTC):", datetime.utcnow().time())
    user_datetime = datetime.combine(user_date, user_time).replace(tzinfo=timezone.utc)

    ascent_rate = st.number_input("Prędkość wznoszenia (m/s):", 1.0, 20.0, 5.0)
    descent_rate = st.number_input("Prędkość opadania (m/s):", -30.0, -1.0, -8.0)
    burst_altitude = st.number_input("Wysokość pęknięcia (m):", 1000, 30000, 30000)

    data_source = st.radio("Źródło danych wiatru:", ["GFS", "ML"], index=0)

# --- Inicjalizacja sesji ---
if "trajectory_df" not in st.session_state:
    st.session_state.trajectory_df = None

# --- Funkcja interpolacji ---
def interpolate_wind(wind_profile, altitude):
    altitudes = wind_profile["Altitude_m"].values
    speeds = wind_profile["WindSpeed_m_s"].values
    dirs = wind_profile["WindDir_deg"].values
    return np.interp(altitude, altitudes, speeds), np.interp(altitude, altitudes, dirs)

# --- Funkcja symulacji ---
def simulate_balloon(lat, lon, ascent_rate, descent_rate, burst_altitude, wind_profile):
    data = []
    current_lat, current_lon = lat, lon
    t = 0  # czas w sekundach od startu

    # Wznoszenie
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

    # Opadanie
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

# --- Symulacja na podstawie wybranego źródła danych ---
if st.button("Uruchom symulację"):
    if data_source == "GFS":
        wind_profile = download_and_parse_gfs_for_window(start_lat, start_lon, user_datetime)
        if wind_profile is not None:
            st.session_state.trajectory_df = simulate_balloon(start_lat, start_lon, ascent_rate, descent_rate, burst_altitude, wind_profile)
            st.success("✅ Symulacja GFS zakończona!")
    elif data_source == "ML":
        try:
            wind_rows = []
            for h in range(1000, burst_altitude + 1000, 1000):
                input_df = load_input_for_ml(user_datetime, start_lat, start_lon, altitude_m=h)
                result = predict_wind_ml(input_df, altitude_m=h)
                wind_rows.append({
                    "Altitude_m": h,
                    "WindSpeed_m_s": result["Predicted_WindSpeed_m_s"],
                    "WindDir_deg": 0  # Brak predykcji kierunku w aktualnym modelu
                })
            wind_profile = pd.DataFrame(wind_rows)
            st.session_state.trajectory_df = simulate_balloon(start_lat, start_lon, ascent_rate, descent_rate, burst_altitude, wind_profile)
            st.success("✅ Symulacja ML zakończona!")
        except Exception as e:
            st.error(f"❌ Błąd ML: {e}")

# --- Wyświetlanie wyników ---
if st.session_state.trajectory_df is not None:
    st.subheader("📊 Wyniki symulacji")
    st.dataframe(st.session_state.trajectory_df)

    csv = st.session_state.trajectory_df.to_csv(index=False).encode("utf-8")
    st.download_button("Pobierz trajektorię (CSV)", data=csv, file_name="trajektoria.csv", mime="text/csv")

    m = folium.Map(location=[start_lat, start_lon], zoom_start=6)
    folium.Marker([start_lat, start_lon], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)

    burst_row = st.session_state.trajectory_df[st.session_state.trajectory_df["Burst"] == "Yes"].iloc[0]
    folium.Marker([
        burst_row["Latitude"], burst_row["Longitude"]
    ], tooltip=f"Pęknięcie ({burst_altitude} m)", icon=folium.Icon(color="blue")).add_to(m)

    folium.PolyLine(st.session_state.trajectory_df[["Latitude", "Longitude"]].values, color="black").add_to(m)
    folium.Marker([
        st.session_state.trajectory_df.iloc[-1]["Latitude"],
        st.session_state.trajectory_df.iloc[-1]["Longitude"]
    ], tooltip="Lądowanie", icon=folium.Icon(color="red")).add_to(m)

    st_folium(m, width=700, height=500, key="mapa")

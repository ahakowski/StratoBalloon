import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone
import tempfile
import xarray as xr
import os

# --- Funkcja pobierania GFS subset ---
def download_gfs_subset(lat, lon, date, run):
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    file = f"gfs.t{run}z.pgrb2.0p25.f000"

    leftlon = lon - 1
    rightlon = lon + 1
    toplat = lat + 1
    bottomlat = lat - 1

    params = {
        "file": file,
        "lev_1000_mb": "on",
        "lev_925_mb": "on",
        "lev_850_mb": "on",
        "lev_700_mb": "on",
        "lev_500_mb": "on",
        "lev_400_mb": "on",
        "lev_300_mb": "on",
        "lev_250_mb": "on",
        "lev_200_mb": "on",
        "lev_150_mb": "on",
        "lev_100_mb": "on",
        "lev_70_mb": "on",
        "lev_50_mb": "on",
        "lev_30_mb": "on",
        "lev_20_mb": "on",
        "lev_10_mb": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "subregion": "",
        "leftlon": leftlon,
        "rightlon": rightlon,
        "toplat": toplat,
        "bottomlat": bottomlat,
        "dir": f"/gfs.{date}/{run}/atmos"
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
        with open(tmpfile.name, "wb") as f:
            f.write(response.content)
        return tmpfile.name
    else:
        st.error(f"Błąd pobierania GFS subset: {response.status_code}")
        return None

# --- Funkcja konwersji ciśnienia na wysokość ---
def pressure_to_altitude(pressure_hpa):
    return round(44330 * (1 - (pressure_hpa / 1013.25)**0.1903))

# --- Parsowanie danych GFS ---
def parse_gfs_data(grib_file, lat, lon, forecast_datetime):
    ds = xr.open_dataset(grib_file, engine="cfgrib")
    u_wind = ds["u"].sel(latitude=lat, longitude=lon, method="nearest")
    v_wind = ds["v"].sel(latitude=lat, longitude=lon, method="nearest")

    profile_data = []
    for i, level in enumerate(u_wind["isobaricInhPa"].values):
        u = float(u_wind.values[i])
        v = float(v_wind.values[i])
        speed = np.sqrt(u**2 + v**2)
        direction = (np.arctan2(-u, -v) * 180 / np.pi) % 360
        profile_data.append({
            "Pressure_hPa": int(level),
            "Altitude_m": pressure_to_altitude(level),
            "WindSpeed_m_s": round(speed, 2),
            "WindDir_deg": round(direction, 1),
            "Forecast_Run": forecast_datetime
        })
    return pd.DataFrame(profile_data).sort_values("Altitude_m")

# --- Interpolacja wiatru ---
def interpolate_wind(wind_profile, altitude):
    altitudes = wind_profile["Altitude_m"].values
    speeds = wind_profile["WindSpeed_m_s"].values
    dirs = wind_profile["WindDir_deg"].values
    return np.interp(altitude, altitudes, speeds), np.interp(altitude, altitudes, dirs)

# --- Symulacja balonu ---
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

# --- Streamlit UI ---
st.title("Symulacja balonu stratosferycznego (NOAA GFS – do 30 km)")

start_lat = st.number_input("Szerokość geograficzna (Start):", -90.0, 90.0, 52.4072)
start_lon = st.number_input("Długość geograficzna (Start):", -180.0, 180.0, 16.9252)

default_date = datetime.now(timezone.utc)
user_date = st.date_input("Data startu:", default_date.date())
user_time = st.time_input("Godzina startu (UTC):", default_date.time())
user_datetime = datetime.combine(user_date, user_time)

ascent_rate = st.number_input("Prędkość wznoszenia (m/s):", 1.0, 20.0, 5.0)
descent_rate = st.number_input("Prędkość opadania (m/s):", -30.0, -1.0, -8.0)
burst_altitude = st.number_input("Wysokość pęknięcia (m):", 1000, 35000, 30000)

if "wind_profile" not in st.session_state:
    st.session_state.wind_profile = None
if "trajectory_df" not in st.session_state:
    st.session_state.trajectory_df = None
if "forecast_info" not in st.session_state:
    st.session_state.forecast_info = None

if st.button("Pobierz dane GFS i uruchom symulację"):
    gfs_date = user_datetime.strftime("%Y%m%d")
    run = "06"  # dla przykładu, można dodać logikę wyboru
    forecast_datetime = f"{user_datetime.strftime('%Y-%m-%d')} {run}:00 UTC"
    grib_file = download_gfs_subset(start_lat, start_lon, gfs_date, run)
    if grib_file:
        st.session_state.wind_profile = parse_gfs_data(grib_file, start_lat, start_lon, forecast_datetime)
        st.session_state.forecast_info = forecast_datetime
        os.remove(grib_file)
        st.session_state.trajectory_df = simulate_balloon(
            start_lat, start_lon, ascent_rate, descent_rate, burst_altitude, st.session_state.wind_profile
        )

# --- Wyświetlanie wyników ---
if st.session_state.wind_profile is not None:
    st.subheader(f"Profil wiatru – dane GFS z: {st.session_state.forecast_info}")
    st.dataframe(st.session_state.wind_profile, key="wind_table")

if st.session_state.trajectory_df is not None:
    st.subheader("Wyniki symulacji")
    st.dataframe(st.session_state.trajectory_df, key="traj_table")

    csv = st.session_state.trajectory_df.to_csv(index=False).encode('utf-8')
    st.download_button("Pobierz trajektorię (CSV)", data=csv, file_name="trajektoria.csv", mime="text/csv")

    m = folium.Map(location=[start_lat, start_lon], zoom_start=6)
    folium.Marker([start_lat, start_lon], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)

    burst_row = st.session_state.trajectory_df[st.session_state.trajectory_df["Burst"] == "Yes"].iloc[0]
    folium.Marker(
        [burst_row["Latitude"], burst_row["Longitude"]],
        tooltip=f"Pęknięcie ({burst_altitude} m)",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    folium.PolyLine(st.session_state.trajectory_df[["Latitude", "Longitude"]].values, color="black").add_to(m)
    folium.Marker([st.session_state.trajectory_df.iloc[-1]["Latitude"],
                   st.session_state.trajectory_df.iloc[-1]["Longitude"]],
                  tooltip="Lądowanie", icon=folium.Icon(color="red")).add_to(m)
    st_folium(m, width=700, height=500, key="balloon_map")

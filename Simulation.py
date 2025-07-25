import requests
import folium
import numpy as np

# --- KONFIGURACJA STARTOWA ---
START_LAT = 52.4072
START_LON = 16.9252
START_TIME = "2025-07-25T12:00:00Z"  # czas startu UTC

# Parametry balonu
ASCENT_RATE = 5.0      # m/s (typowe dla balonu z 1.5 kg ładunku)
DESCENT_RATE = -8.0    # m/s (opadanie ze spadochronem)
BURST_ALTITUDE = 30000 # pęknięcie na 30 km

# --- FUNKCJE ---

def get_wind_profile(lat, lon):
    """
    Pobiera profil wiatru (wysokość, prędkość, kierunek).
    Open-Meteo nie daje profilu pionowego dla wiatru, więc bierzemy dane
    i rozciągamy je w uproszczony sposób. Można to zastąpić danymi GFS.
    """
    # Pobieramy dane wiatru na 100m
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=windspeed_100m,winddirection_100m&start={START_TIME}&end={START_TIME}&timezone=UTC"
    r = requests.get(url)
    data = r.json()
    wind_speed = data['hourly']['windspeed_100m'][0]  # km/h
    wind_dir = data['hourly']['winddirection_100m'][0]  # stopnie
    # Zakładamy podobny kierunek i prędkość w górze (proste założenie)
    profile = [(h, wind_speed, wind_dir) for h in range(0, 31000, 1000)]
    return profile

def simulate_flight(lat, lon, ascent_rate, descent_rate, burst_altitude):
    """
    Symuluje wznoszenie i opadanie balonu, zwraca punkty trajektorii.
    """
    wind_profile = get_wind_profile(lat, lon)
    points = [(lat, lon, 0)]  # (lat, lon, wysokość)
    current_lat, current_lon = lat, lon

    # Wznoszenie
    for h in range(0, burst_altitude + 100, 100):
        wind_speed, wind_dir = wind_profile[min(h // 1000, len(wind_profile)-1)][1:]
        wind_dir_rad = np.deg2rad(270 - wind_dir)
        dt = 100 / ascent_rate
        dx = (wind_speed / 3.6) * dt * np.cos(wind_dir_rad) / 111000
        dy = (wind_speed / 3.6) * dt * np.sin(wind_dir_rad) / 111000
        current_lat += dy
        current_lon += dx
        points.append((current_lat, current_lon, h))

    burst_point = (current_lat, current_lon, burst_altitude)

    # Opadanie
    for h in range(burst_altitude, 0, -100):
        wind_speed, wind_dir = wind_profile[min(h // 1000, len(wind_profile)-1)][1:]
        wind_dir_rad = np.deg2rad(270 - wind_dir)
        dt = 100 / abs(descent_rate)
        dx = (wind_speed / 3.6) * dt * np.cos(wind_dir_rad) / 111000
        dy = (wind_speed / 3.6) * dt * np.sin(wind_dir_rad) / 111000
        current_lat += dy
        current_lon += dx
        points.append((current_lat, current_lon, h))

    landing_point = (current_lat, current_lon, 0)
    return points, burst_point, landing_point

def generate_map(points, burst_point, landing_point):
    """
    Tworzy interaktywną mapę HTML z trajektorią balonu.
    """
    start_point = points[0]
    m = folium.Map(location=[start_point[0], start_point[1]], zoom_start=7)

    # Markery
    folium.Marker([start_point[0], start_point[1]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([burst_point[0], burst_point[1]], tooltip=f"Pęknięcie ({burst_point[2]/1000:.1f} km)", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker([landing_point[0], landing_point[1]], tooltip="Lądowanie", icon=folium.Icon(color="red")).add_to(m)

    # Linia trasy
    line_points = [(p[0], p[1]) for p in points]
    folium.PolyLine(line_points, color="black", weight=2).add_to(m)

    m.save("balloon_flight_map.html")
    print("Mapa zapisana do: balloon_flight_map.html")

# --- GŁÓWNY PROGRAM ---
if __name__ == "__main__":
    points, burst_point, landing_point = simulate_flight(START_LAT, START_LON, ASCENT_RATE, DESCENT_RATE, BURST_ALTITUDE)
    generate_map(points, burst_point, landing_point)
    print("Start:", points[0])
    print("Pęknięcie:", burst_point)
    print("Lądowanie:", landing_point)

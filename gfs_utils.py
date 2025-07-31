import os
import requests
import pandas as pd
import numpy as np
import xarray as xr
import tempfile
from datetime import datetime, timedelta
# Założenie: masz zdefiniowany słownik pressure_to_height
pressure_to_height = {
    1000: 0.1, 925: 0.7, 850: 1.5, 700: 3.0, 500: 5.5, 400: 7.2, 300: 9.5,
    250: 10.8, 200: 12.0, 150: 13.5, 100: 16.0, 70: 18.5, 50: 20.5,
    30: 23.0, 20: 26.0, 10: 30.0
}

def get_gfs_data_for_all_levels(lat, lon, target_date, window_days=7):
    """
    Automatyczne pobieranie danych GFS z cache/dysku lub z internetu,
    dla każdego poziomu ciśnienia.
    """
    wind_profiles_by_level = {}

    for pressure in pressure_to_height:
        try:
            df = download_and_parse_gfs_for_window(
                lat=lat,
                lon=lon,
                end_datetime=target_date,
                pressure_hpa=pressure,
                window_days=window_days
            )
            wind_profiles_by_level[pressure] = df
        except Exception as e:
            print(f"⚠️ Błąd przy pobieraniu danych GFS dla {pressure} hPa: {e}")

    return wind_profiles_by_level

# Zwraca poziom ciśnienia najbliższy podanej wysokości (w metrach)
def closest_pressure_for_altitude(altitude_m):
    height_km = altitude_m / 1000
    return min(pressure_to_height.keys(), key=lambda p: abs(pressure_to_height[p] - height_km))

def load_input_for_ml(target_date, lat, lon, altitude_m=None, pressure_hpa=None):
    """
    Ładuje dane GFS z ostatnich 7 dni dla wskazanej lokalizacji i wysokości/ciśnienia.
    Zwraca DataFrame gotowy do wejścia dla modelu ML.
    """
    assert altitude_m or pressure_hpa, "Podaj 'altitude_m' lub 'pressure_hpa'"

    if altitude_m is not None:
        pressure_hpa = closest_pressure_for_altitude(altitude_m)

    df = download_and_parse_gfs_for_window(lat, lon, target_date, pressure_hpa=pressure_hpa, window_days=7)
    return df
def pressure_to_altitude(pressure_hpa):
    return round(44330 * (1 - (pressure_hpa / 1013.25)**0.1903))

def download_gfs_file(lat, lon, date, run_hour="06"):
    """Pobiera plik GFS dla wskazanej daty i lokalizacji."""
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    file = f"gfs.t{run_hour}z.pgrb2.0p25.f000"

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
        "dir": f"/gfs.{date.strftime('%Y%m%d')}/{run_hour}/atmos"
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
        with open(tmpfile.name, "wb") as f:
            f.write(response.content)
        return tmpfile.name
    else:
        raise Exception(f"Błąd pobierania GFS ({date}): {response.status_code}")

def parse_gfs(grib_file, lat, lon, pressure_hpa):
    """Parsuje dane z GRIB i wybiera wiatr na zadanym poziomie."""
    ds = xr.open_dataset(grib_file, engine="cfgrib")
    u = ds["u"].sel(latitude=lat, longitude=lon, isobaricInhPa=pressure_hpa, method="nearest").values
    v = ds["v"].sel(latitude=lat, longitude=lon, isobaricInhPa=pressure_hpa, method="nearest").values

    speed = np.sqrt(u**2 + v**2)
    direction = (np.arctan2(-u, -v) * 180 / np.pi) % 360

    return float(speed), float(direction)

def download_and_parse_gfs_for_window(lat, lon, end_datetime, pressure_hpa, window_days=7):
    """
    Pobiera dane GFS z ostatnich dni (lub do wskazanej daty) do predykcji ML.
    Zwraca DataFrame z kolumnami: Date, WindSpeed_m_s, WindDir_deg
    """
    df = []
    for i in range(window_days, 0, -1):
        date = end_datetime - timedelta(days=i)
        cache_path = f"gfs_cache/gfs_{pressure_hpa}hPa_{date.strftime('%Y%m%d')}.csv"
        if os.path.exists(cache_path):
            day_df = pd.read_csv(cache_path)
        else:
            try:
                grib_file = download_gfs_file(lat, lon, date)
                speed, direction = parse_gfs(grib_file, lat, lon, pressure_hpa)
                os.remove(grib_file)
                day_df = pd.DataFrame([{
                    "Date": date.strftime("%Y-%m-%d"),
                    "WindSpeed_m_s": speed,
                    "WindDir_deg": direction
                }])
                os.makedirs("gfs_cache", exist_ok=True)
                day_df.to_csv(cache_path, index=False)
            except Exception as e:
                print(f"⚠️ Błąd dla {date}: {e}")
                continue
        df.append(day_df)

    result_df = pd.concat(df).reset_index(drop=True)
    result_df["Date"] = pd.to_datetime(result_df["Date"])
    return result_df

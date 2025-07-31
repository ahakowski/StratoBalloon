
import pandas as pd
import numpy as np
import xarray as xr
import requests
import tempfile
import os
from datetime import datetime, timedelta

# Konfiguracja obszaru (300x300 km wokół Poznania)
center_lat = 52.4
center_lon = 16.9
delta_deg = 2.7
toplat = center_lat + delta_deg
bottomlat = center_lat - delta_deg
leftlon = center_lon - delta_deg
rightlon = center_lon + delta_deg

# Poziomy ciśnienia do 10 hPa (~30 km)
target_levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]

# Przygotowanie pustej listy do zbierania danych
all_data = []

# Bierzemy dane z 30 dni wstecz od dziś
today = datetime.utcnow()
dates = [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(30)]

# Przetwarzanie tylko runów 00Z i f000 (analiza)
for date in dates:
    year = date[:4]
    url = f"https://data.rda.ucar.edu/ds084.1/{year}/{date}/gfs.0p25.{date}00.f000.grib2"
    print(f"Pobieranie: {url}")

    try:
        # Pobieranie pliku
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            print(f"Błąd pobierania: {response.status_code}")
            continue
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
        with open(tmpfile.name, "wb") as f:
            f.write(response.content)

        # Wczytanie danych z cfgrib
        ds = xr.open_dataset(tmpfile.name, engine="cfgrib")
        ds = ds.sel(isobaricInhPa=[lvl for lvl in target_levels if lvl in ds.isobaricInhPa.values])
        ds = ds.sel(latitude=slice(toplat, bottomlat), longitude=slice(leftlon, rightlon))

        # Ekstrakcja u i v
        u = ds['u']
        v = ds['v']

        speed = np.sqrt(u**2 + v**2)
        direction = (np.arctan2(-u, -v) * 180 / np.pi) % 360

        df = pd.DataFrame({
            'Date': date,
            'Latitude': speed.latitude.values.repeat(len(speed.longitude) * len(speed.isobaricInhPa)),
            'Longitude': np.tile(speed.longitude.values, len(speed.latitude) * len(speed.isobaricInhPa)),
            'Pressure_hPa': np.repeat(speed.isobaricInhPa.values, len(speed.latitude) * len(speed.longitude)),
            'WindSpeed_m_s': speed.values.flatten(),
            'WindDir_deg': direction.values.flatten()
        })
        all_data.append(df)

        os.remove(tmpfile.name)
    except Exception as e:
        print(f"Błąd przetwarzania {date}: {e}")
        continue

# Łączenie wszystkich danych i zapis
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("gfs_data.csv", index=False)
print("Zapisano dane do gfs_data.csv")

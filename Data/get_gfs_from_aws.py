
import pandas as pd
import numpy as np
import xarray as xr
import requests
import tempfile
import os
import warnings
from datetime import datetime, timedelta
from cfgrib import open_datasets

warnings.filterwarnings("ignore", category=FutureWarning)

# 📍 Centrum: Poznań
center_lat = 52.4
center_lon = 16.9
delta_deg = 2.7  # ≈ 300 km

toplat = center_lat + delta_deg
bottomlat = center_lat - delta_deg
leftlon = center_lon - delta_deg
rightlon = center_lon + delta_deg

target_levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]

today = datetime.utcnow()
dates = [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(90)]

all_data = []

for date in dates:
    url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/00/atmos/gfs.t00z.pgrb2.0p25.f000"
    print(f"🔽 Pobieranie: {url}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmpfile:
            tmpfile.write(requests.get(url, timeout=60).content)
            tmpfile_path = tmpfile.name

        # Otwórz z osobnego pliku po zamknięciu tmpfile
        datasets = open_datasets(tmpfile_path, filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
        ds = next(ds for ds in datasets if 'u' in ds.variables and 'v' in ds.variables and 'isobaricInhPa' in ds.coords)

        ds = ds.sel(isobaricInhPa=[lvl for lvl in target_levels if lvl in ds.isobaricInhPa.values])
        ds = ds.sel(latitude=slice(toplat, bottomlat), longitude=slice(leftlon, rightlon))

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

        for d in datasets:
            d.close()
        os.remove(tmpfile_path)

    except Exception as e:
        print(f"⚠️ Błąd: {e}")
        continue

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("gfs_data.csv", index=False)
    print("✅ Dane zapisane do gfs_data.csv")
else:
    print("❗ Brak danych do zapisania.")

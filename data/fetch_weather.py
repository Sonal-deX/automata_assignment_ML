"""
Download real historical daily weather for 12 Sri Lankan cities.

Source : Open-Meteo Historical Weather API (https://open-meteo.com/en/docs/historical-weather-api)
         which serves the ECMWF ERA5 / ERA5-Land reanalysis (real observations assimilated
         into a physical model, ~9-25 km grid). Free, no API key.
Period : 2015-01-01 to 2024-12-31 (10 years, daily)

Run:  python data/fetch_weather.py
Output: data/raw/sri_lanka_weather_daily.csv
"""
import time
from pathlib import Path

import pandas as pd
import requests

# city -> (latitude, longitude, climatic zone)
# Zones follow the Department of Meteorology Sri Lanka classification (Wet / Intermediate / Dry)
CITIES = {
    "Colombo":      (6.9271, 79.8612, "Wet"),
    "Galle":        (6.0535, 80.2210, "Wet"),
    "Ratnapura":    (6.6828, 80.3992, "Wet"),
    "Kandy":        (7.2906, 80.6337, "Wet"),
    "Nuwara Eliya": (6.9497, 80.7891, "Wet"),
    "Kurunegala":   (7.4863, 80.3647, "Intermediate"),
    "Badulla":      (6.9934, 81.0550, "Intermediate"),
    "Anuradhapura": (8.3114, 80.4037, "Dry"),
    "Jaffna":       (9.6615, 80.0255, "Dry"),
    "Trincomalee":  (8.5874, 81.2152, "Dry"),
    "Batticaloa":   (7.7310, 81.6747, "Dry"),
    "Hambantota":   (6.1241, 81.1185, "Dry"),
}

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean", "pressure_msl_mean", "cloud_cover_mean",
    "dew_point_2m_mean",
]

URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2015-01-01", "2024-12-31"


def fetch_city(name, lat, lon, zone):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Colombo",
    }
    for attempt in range(8):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            break
        # free tier is rate limited per minute -> wait a full minute and retry
        print(f"  {name}: HTTP {r.status_code}, waiting 65s and retrying...")
        time.sleep(65)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js["daily"])
    df.insert(0, "city", name)
    df.insert(1, "zone", zone)
    df.insert(2, "latitude", lat)
    df.insert(3, "longitude", lon)
    df.insert(4, "elevation", js.get("elevation"))
    return df


def main():
    out = Path(__file__).parent / "raw"
    cache = out / "per_city"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (lat, lon, zone) in CITIES.items():
        f = cache / f"{name.replace(' ', '_')}.csv"
        if f.exists():  # already downloaded
            frames.append(pd.read_csv(f))
            continue
        print("Downloading", name)
        df_city = fetch_city(name, lat, lon, zone)
        df_city.to_csv(f, index=False)
        frames.append(df_city)
        time.sleep(15)  # be polite to the free API
    df = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    df.to_csv(out / "sri_lanka_weather_daily.csv", index=False)
    print("Saved", df.shape, "->", out / "sri_lanka_weather_daily.csv")


if __name__ == "__main__":
    main()

import requests
from app.db import supabase

WEATHER_VARS = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation"
]


def fetch_sites():
    result = supabase.table("sites").select("*").eq("active", True).execute()
    return result.data


def fetch_weather(site):
    params = {
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "hourly": ",".join(WEATHER_VARS),
        "forecast_days": 2,
        "timezone": "Europe/Amsterdam"
    }

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params
    )

    return response.json()


def save_weather(site_id, payload):
    hourly = payload["hourly"]

    rows = []

    for i, ts in enumerate(hourly["time"]):
        rows.append({
            "site_id": site_id,
            "timestamp": ts,
            "temperature": hourly["temperature_2m"][i],
            "cloud_cover": hourly["cloud_cover"][i],
            "radiation": hourly["shortwave_radiation"][i]
        })

    supabase.table("weather_forecast_hourly").upsert(rows).execute()


def sync_weather_for_all_sites():
    for site in fetch_sites():
        data = fetch_weather(site)
        save_weather(site["id"], data)

from app.db import supabase
import numpy as np


def run_forecast_for_site(site_id: str):

    measurements = (
        supabase.table("site_measurements_15m")
        .select("*")
        .eq("site_id", site_id)
        .order("timestamp", desc=True)
        .limit(96)
        .execute()
    )

    if not measurements.data:
        return {"error": "no measurements"}

    load_values = [m["load_kw"] for m in measurements.data]

    avg_load = np.mean(load_values)

    forecast = []

    for i in range(96):
        forecast.append({
            "site_id": site_id,
            "quarter": i,
            "pred_load_kw": float(avg_load)
        })

    supabase.table("forecast_predictions_15m").upsert(forecast).execute()

    return {
        "site_id": site_id,
        "status": "forecast_created",
        "avg_load": float(avg_load)
    }

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from app.db import supabase

MODEL_CACHE = {}


def _fetch_measurements(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    result = (
        supabase.table("site_measurements_15m")
        .select("*")
        .eq("site_id", site_id)
        .gte("ts_utc", start_ts.isoformat())
        .lte("ts_utc", end_ts.isoformat())
        .order("ts_utc")
        .execute()
    )
    df = pd.DataFrame(result.data or [])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def _fetch_weather_forecast(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    result = (
        supabase.table("weather_forecast_hourly")
        .select("*")
        .eq("site_id", site_id)
        .gte("ts_utc", start_ts.isoformat())
        .lte("ts_utc", end_ts.isoformat())
        .order("ts_utc")
        .execute()
    )
    df = pd.DataFrame(result.data or [])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def _quarter_index(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 4 + (ts.dt.minute // 15)


def _similar_day_load_forecast(df: pd.DataFrame) -> np.ndarray:
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Gebruik laatste 4 weken als mogelijk
    if len(df) >= 96 * 28:
        last_4_weeks = df["power_kw"].tail(96 * 28).values.reshape(28, 96)
        return np.mean(last_4_weeks, axis=0)

    # fallback: laatste 7 dagen
    if len(df) >= 96 * 7:
        last_week = df["power_kw"].tail(96 * 7).values.reshape(7, 96)
        return np.mean(last_week, axis=0)

    # fallback: laatste dag
    if len(df) >= 96:
        return df["power_kw"].tail(96).values

    # fallback: gemiddelde
    avg = df["power_kw"].mean() if len(df) > 0 else 0.0
    return np.full(96, avg)


def _load_confidence(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.sort_values("ts_utc").reset_index(drop=True)

    if len(df) >= 96 * 7:
        values = df["power_kw"].tail(96 * 7).values.reshape(7, 96)
        center = np.mean(values, axis=0)
        spread = np.std(values, axis=0)
    else:
        center = _similar_day_load_forecast(df)
        spread = np.full(96, np.std(df["power_kw"].values) if len(df) > 10 else 0.25)

    lower = center - spread
    upper = center + spread
    return lower, upper


def _estimate_pv_from_weather(weather_df: pd.DataFrame, site_metadata: dict | None) -> np.ndarray:
    """
    Eenvoudige eerste PV forecast:
    op basis van GHI en eventueel pv_capacity uit metadata.
    """
    if weather_df.empty:
        return np.zeros(96)

    pv_capacity_kwp = 0.0
    if site_metadata:
        pv_capacity_kwp = float(site_metadata.get("pv_capacity_kwp", 0.0) or 0.0)

    # Als geen capacity bekend is, schat PV voorlopig op 0
    if pv_capacity_kwp <= 0:
        return np.zeros(96)

    # Hourly GHI naar kwartieren
    ghi = weather_df["ghi_wm2"].fillna(0).values
    ghi_q = np.repeat(ghi, 4)[:96]

    # Eenvoudige omzetting:
    # bij 1000 W/m2 ongeveer rond nominale piek
    pv_kw = pv_capacity_kwp * (ghi_q / 1000.0)

    # begrens
    pv_kw = np.clip(pv_kw, 0, pv_capacity_kwp)
    return pv_kw


def _estimate_ev_baseline(df: pd.DataFrame) -> np.ndarray:
    if len(df) >= 96 * 7:
        ev = df["metadata"].apply(lambda x: x.get("ev_kw", 0) if isinstance(x, dict) else 0).values
        ev = ev[-96 * 7:].reshape(7, 96)
        return np.mean(ev, axis=0)

    return np.zeros(96)


def _fetch_latest_metadata(site_id: str):
    result = (
        supabase.table("site_measurements_15m")
        .select("metadata")
        .eq("site_id", site_id)
        .order("ts_utc", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        return None
    return rows[0].get("metadata")


def _save_forecast(site_id: str, load_fc: np.ndarray, pv_fc: np.ndarray, ev_fc: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray, model_id: str):
    now = datetime.now(timezone.utc)
    start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)

    rows = []
    for i in range(96):
        ts = start + timedelta(minutes=15 * i)
        net_fc = float(load_fc[i] - pv_fc[i] + ev_fc[i])

        rows.append({
            "site_id": site_id,
            "ts_utc": ts.isoformat(),
            "model_id": model_id,
            "predicted_power_kw": net_fc,          # legacy compat
            "predicted_load_kw": float(load_fc[i]),
            "predicted_pv_kw": float(pv_fc[i]),
            "predicted_ev_kw": float(ev_fc[i]),
            "predicted_net_kw": net_fc,
            "confidence_lower": float(lower[i]),
            "confidence_upper": float(upper[i]),
            "forecast_type": "day_ahead",
            "metadata": {
                "source": "enersim-api",
                "version": model_id
            }
        })

    supabase.table("forecast_predictions_15m").upsert(rows).execute()


def train_models_for_site(site_id: str):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365)

    df = _fetch_measurements(site_id, start, now)
    if df.empty:
        return {"error": "geen meetdata gevonden"}

    MODEL_CACHE[site_id] = {
        "status": "ready",
        "trained_at": now.isoformat(),
        "rows_used": len(df),
        "model_id": "similar_day_plus_weather_v1",
    }

    return {
        "site_id": site_id,
        "status": "trained",
        "rows_used": len(df),
        "model_id": "similar_day_plus_weather_v1",
    }


def _ensure_models(site_id: str):
    if site_id not in MODEL_CACHE:
        train_models_for_site(site_id)


def run_forecast_for_site(site_id: str):
    _ensure_models(site_id)

    now = datetime.now(timezone.utc)
    hist_start = now - timedelta(days=60)
    horizon_end = now + timedelta(hours=24)

    measurements = _fetch_measurements(site_id, hist_start, now)
    if measurements.empty:
        return {"error": "geen meetdata gevonden"}

    weather_fc = _fetch_weather_forecast(site_id, now - timedelta(hours=1), horizon_end)
    if weather_fc.empty:
        return {"error": "geen weather forecast gevonden"}

    metadata = _fetch_latest_metadata(site_id)

    load_fc = _similar_day_load_forecast(measurements)
    lower, upper = _load_confidence(measurements)
    pv_fc = _estimate_pv_from_weather(weather_fc, metadata)
    ev_fc = _estimate_ev_baseline(measurements)

    model_id = MODEL_CACHE[site_id]["model_id"]
    _save_forecast(site_id, load_fc, pv_fc, ev_fc, lower, upper, model_id)

    return {
        "site_id": site_id,
        "status": "forecast_created",
        "model_id": model_id,
        "rows_written": 96,
        "sample": [
            {
                "predicted_load_kw": float(load_fc[0]),
                "predicted_pv_kw": float(pv_fc[0]),
                "predicted_ev_kw": float(ev_fc[0]),
                "predicted_net_kw": float(load_fc[0] - pv_fc[0] + ev_fc[0]),
            }
        ]
    }

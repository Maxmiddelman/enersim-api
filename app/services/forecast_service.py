import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from app.db import supabase

MODEL_CACHE = {}

def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value to a JSON-safe float. Replaces NaN/Infinity with default."""
    try:
        f = float(value)
        if math.isfinite(f):
            return round(f, 4)
        return default
    except (TypeError, ValueError):
        return default

def _sanitize_array(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Replace all NaN and Infinity values in an array with default."""
    arr = np.where(np.isfinite(arr), arr, default)
    return arr

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
        for col in ["power_kw", "load_kw", "pv_kw", "ev_kw"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
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
        for col in ["ghi_wm2", "temperature_c", "cloud_cover_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def _quarter_index(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 4 + (ts.dt.minute // 15)

def _similar_day_load_forecast(df: pd.DataFrame) -> np.ndarray:
    df = df.sort_values("ts_utc").reset_index(drop=True)

    if "power_kw" not in df.columns or df["power_kw"].isna().all():
        return np.zeros(96)

    values = df["power_kw"].dropna().values

    if len(values) == 0:
        return np.zeros(96)

    if len(values) >= 96 * 28:
        n = 96 * 28
        trimmed = values[-n:]
        daily = trimmed.reshape(28, 96)
        result = np.nanmean(daily, axis=0)
    elif len(values) >= 96 * 7:
        n = 96 * 7
        trimmed = values[-n:]
        daily = trimmed.reshape(7, 96)
        result = np.nanmean(daily, axis=0)
    elif len(values) >= 96:
        result = values[-96:]
    else:
        avg = np.nanmean(values) if len(values) > 0 else 0.0
        result = np.full(96, _safe_float(avg))

    return _sanitize_array(result)

def _load_confidence(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.sort_values("ts_utc").reset_index(drop=True)

    if "power_kw" not in df.columns or df["power_kw"].isna().all():
        center = np.zeros(96)
        lower = np.full(96, -0.25)
        upper = np.full(96, 0.25)
        return lower, upper

    values = df["power_kw"].dropna().values

    if len(values) >= 96 * 7:
        n = 96 * 7
        trimmed = values[-n:]
        daily = trimmed.reshape(7, 96)
        center = np.nanmean(daily, axis=0)
        spread = np.nanstd(daily, axis=0)
        spread = np.where(np.isfinite(spread) & (spread > 0.1), spread, 0.25)
    else:
        center = _similar_day_load_forecast(df)
        std_val = np.nanstd(values) if len(values) > 10 else 0.25
        std_val = _safe_float(std_val, 0.25)
        if std_val < 0.1:
            std_val = 0.25
        spread = np.full(96, std_val)

    center = _sanitize_array(center)
    spread = _sanitize_array(spread, 0.25)

    lower = center - spread
    upper = center + spread
    return _sanitize_array(lower), _sanitize_array(upper)

def _estimate_pv_from_weather(weather_df: pd.DataFrame, site_metadata: dict | None) -> np.ndarray:
    if weather_df.empty:
        return np.zeros(96)

    pv_capacity_kwp = 0.0
    if site_metadata and isinstance(site_metadata, dict):
        try:
            pv_capacity_kwp = float(site_metadata.get("pv_capacity_kwp", 0.0) or 0.0)
        except (TypeError, ValueError):
            pv_capacity_kwp = 0.0

    if not math.isfinite(pv_capacity_kwp) or pv_capacity_kwp <= 0:
        return np.zeros(96)

    ghi = weather_df["ghi_wm2"].fillna(0).values
    ghi = np.where(np.isfinite(ghi), ghi, 0)
    ghi_q = np.repeat(ghi, 4)[:96]

    if len(ghi_q) < 96:
        ghi_q = np.pad(ghi_q, (0, 96 - len(ghi_q)), constant_values=0)

    pv_kw = pv_capacity_kwp * (ghi_q / 1000.0)
    pv_kw = np.clip(pv_kw, 0, pv_capacity_kwp)

    return _sanitize_array(pv_kw)

def _estimate_ev_baseline(df: pd.DataFrame) -> np.ndarray:
    if "metadata" not in df.columns:
        return np.zeros(96)

    try:
        values = df["metadata"].dropna().values
        if len(values) < 96 * 7:
            return np.zeros(96)

        ev_values = []
        for m in values[-96 * 7:]:
            if isinstance(m, dict):
                val = m.get("ev_kw", 0)
                ev_values.append(_safe_float(val))
            else:
                ev_values.append(0.0)

        ev_arr = np.array(ev_values)
        if len(ev_arr) == 96 * 7:
            daily = ev_arr.reshape(7, 96)
            result = np.nanmean(daily, axis=0)
            return _sanitize_array(result)
    except Exception:
        pass

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
    meta = rows[0].get("metadata")
    if isinstance(meta, dict):
        return meta
    return None

def _save_forecast(site_id: str, load_fc: np.ndarray, pv_fc: np.ndarray, ev_fc: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray, model_id: str):
    now = datetime.now(timezone.utc)
    start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)

    rows = []
    for i in range(96):
        ts = start + timedelta(minutes=15 * i)
        load_val = _safe_float(load_fc[i])
        pv_val = _safe_float(pv_fc[i])
        ev_val = _safe_float(ev_fc[i])
        net_val = _safe_float(load_val - pv_val + ev_val)
        lower_val = _safe_float(lower[i])
        upper_val = _safe_float(upper[i])

        rows.append({
            "site_id": site_id,
            "ts_utc": ts.isoformat(),
            "model_id": model_id,
            "predicted_power_kw": net_val,
            "predicted_load_kw": load_val,
            "predicted_pv_kw": pv_val,
            "predicted_ev_kw": ev_val,
            "predicted_net_kw": net_val,
            "confidence_lower": lower_val,
            "confidence_upper": upper_val,
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
        return {"error": "geen meetdata gevonden", "site_id": site_id}

    weather_fc = _fetch_weather_forecast(site_id, now - timedelta(hours=1), horizon_end)

    metadata = _fetch_latest_metadata(site_id)

    load_fc = _similar_day_load_forecast(measurements)
    lower, upper = _load_confidence(measurements)

    if weather_fc.empty:
        pv_fc = np.zeros(96)
    else:
        pv_fc = _estimate_pv_from_weather(weather_fc, metadata)

    ev_fc = _estimate_ev_baseline(measurements)

    # Final sanitize pass
    load_fc = _sanitize_array(load_fc)
    pv_fc = _sanitize_array(pv_fc)
    ev_fc = _sanitize_array(ev_fc)
    lower = _sanitize_array(lower)
    upper = _sanitize_array(upper)

    model_id = MODEL_CACHE.get(site_id, {}).get("model_id", "similar_day_plus_weather_v1")

    _save_forecast(site_id, load_fc, pv_fc, ev_fc, lower, upper, model_id)

    return {
        "site_id": site_id,
        "status": "forecast_created",
        "model_id": model_id,
        "rows_written": 96,
        "weather_available": not weather_fc.empty,
        "sample": [
            {
                "predicted_load_kw": _safe_float(load_fc[0]),
                "predicted_pv_kw": _safe_float(pv_fc[0]),
                "predicted_ev_kw": _safe_float(ev_fc[0]),
                "predicted_net_kw": _safe_float(load_fc[0] - pv_fc[0] + ev_fc[0]),
            }
        ]
    }

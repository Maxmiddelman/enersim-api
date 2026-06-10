"""
Verbeterde forecast service voor EnerSim
=======================================

Deze module bevat een verbeterde implementatie van de `forecast_service` voor
EnerSim.  De belangrijkste verandering is dat de load‐forecast niet langer
gebaseerd is op een eenvoudig vergelijkbare‐dag‐gemiddelde, maar op een
getraind LSTM‑model met aandacht.  Het model wordt per locatie getraind op de
historische kwartierdata en gebruikt vervolgens iteratieve voorspellingen
om de dag‐ahead load te bepalen.  PV‑ en EV‑voorspellingen blijven
gebaseerd op de bestaande heuristieken.

Gebruik deze module als alternatief voor `app/services/forecast_service.py`.
Je kunt de functies rechtstreeks importeren in je FastAPI of andere
service-logica.  Voordat je een voorspelling voor een site kunt maken,
dient het model voor die site getraind te zijn via `train_models_for_site`.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from app.db import supabase
from app.models.improved_forecast import (
    DataProcessor,
    LSTMWithAttention,
    train_model,
    predict_future,
)


# Cache om per site model, scaler en processor op te slaan
MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Converteer een waarde naar een float en vervang NaN/Inf door default."""
    try:
        f = float(value)
        if math.isfinite(f):
            return round(f, 4)
        return default
    except (TypeError, ValueError):
        return default


def _sanitize_array(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Vervang NaN en Inf in een numpy-array door een default-waarde."""
    arr = np.where(np.isfinite(arr), arr, default)
    return arr


def _fetch_measurements(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    """Haal kwartierdata op uit Supabase voor een locatie en tijdsinterval."""
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
        # Converteren van relevante kolommen naar numeriek
        for col in ["power_kw", "load_kw", "pv_kw", "ev_kw"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_weather_forecast(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    """Haal uurlijkse weather-forecast op uit Supabase voor een locatie en tijdsinterval."""
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


def _estimate_pv_from_weather(weather_df: pd.DataFrame, site_metadata: Dict[str, Any] | None) -> np.ndarray:
    """Schat PV-vermogen op basis van GHI en geïnstalleerd PV-vermogen."""
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
    # Uurlijkse GHI converteren naar kwartierwaarden
    ghi = weather_df["ghi_wm2"].fillna(0).values
    ghi = np.where(np.isfinite(ghi), ghi, 0)
    ghi_q = np.repeat(ghi, 4)[:96]
    if len(ghi_q) < 96:
        ghi_q = np.pad(ghi_q, (0, 96 - len(ghi_q)), constant_values=0)
    pv_kw = pv_capacity_kwp * (ghi_q / 1000.0)
    pv_kw = np.clip(pv_kw, 0, pv_capacity_kwp)
    return _sanitize_array(pv_kw)


def _estimate_ev_baseline(measurements: pd.DataFrame) -> np.ndarray:
    """Schat EV-baseline op basis van metadata in meetdata."""
    if "metadata" not in measurements.columns:
        return np.zeros(96)
    try:
        values = measurements["metadata"].dropna().values
        if len(values) < 96 * 7:
            return np.zeros(96)
        ev_values: List[float] = []
        for m in values[-96 * 7:]:
            if isinstance(m, dict):
                ev_kw = m.get("ev_kw", 0.0) or 0.0
                ev_values.append(float(ev_kw))
        if len(ev_values) < 96:
            return np.zeros(96)
        daily = np.array(ev_values[-96 * 7:]).reshape(7, 96)
        baseline = np.nanmean(daily, axis=0)
        return _sanitize_array(baseline)
    except Exception:
        return np.zeros(96)


def _fallback_load_forecast(measurements: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maak een eenvoudige load-voorspelling wanneer er onvoldoende data is voor het LSTM-model.

    De fallback gebruikt een gemiddeld verbruiksprofiel per kwartier van de dag.
    Als er te weinig data of geen kolom `power_kw` is, wordt een vlakke nulvoorspelling
    teruggegeven.  Voor een eenvoudige confidence interval wordt één standaardafwijking
    boven en onder het gemiddelde genomen; ontbrekende kwartieren krijgen het globale
    gemiddelde en standaardafwijking.

    Parameters
    ----------
    measurements : pd.DataFrame
        Historische kwartierdata met kolommen `ts_utc` (tijdstempel) en
        `power_kw` (vermogen).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Een tuple van (load_fc, lower, upper) met lengte 96 (dag ahead in
        kwartieren).
    """
    if measurements.empty or 'power_kw' not in measurements.columns:
        load_fc = np.zeros(96)
        return load_fc, load_fc.copy(), load_fc.copy()
    df = measurements.copy()
    # Zorg dat de tijdstempel een datetime is
    df['ts_utc'] = pd.to_datetime(df['ts_utc'], utc=True)
    # Bereken kwartierindex binnen de dag (0-95)
    df['quarter_index'] = df['ts_utc'].dt.hour * 4 + df['ts_utc'].dt.minute // 15
    grouped = df.groupby('quarter_index')['power_kw']
    mean_per_q = grouped.mean()
    std_per_q = grouped.std().fillna(0)
    # Gebruik globale gemiddelde en std voor ontbrekende kwartieren
    global_mean = mean_per_q.mean() if not mean_per_q.empty else 0.0
    global_std = std_per_q.mean() if not std_per_q.empty else 0.0
    load_fc = np.zeros(96)
    lower = np.zeros(96)
    upper = np.zeros(96)
    for q in range(96):
        m = mean_per_q.get(q, global_mean)
        s = std_per_q.get(q, global_std)
        load_fc[q] = m
        lower[q] = max(m - s, 0.0)
        upper[q] = m + s
    return _sanitize_array(load_fc), _sanitize_array(lower), _sanitize_array(upper)


def train_models_for_site(site_id: str) -> Dict[str, Any]:
    """Train het load-voorspellingsmodel voor een locatie en sla op in de cache."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365)
    # Haal tot een jaar aan metingen op
    df = _fetch_measurements(site_id, start, now)
    if df.empty:
        return {"error": "geen meetdata gevonden"}
    # DataProcessor aanmaken; configureer lags en eventuele weerkolommen
    processor = DataProcessor(freq='15min', lags=[1, 4, 8, 12])
    # Hernoem kolommen zodat DataProcessor ze herkent
    df_local = df.rename(columns={'ts_utc': 'timestamp', 'power_kw': 'value'})
    df_prepared = processor.prepare_features(df_local)
    # Train model en scaler; input_window=96 -> 24 uur input, output_window=1 -> 1 kwartier vooruit
    model, scaler = train_model(df_prepared, input_window=96, output_window=1, epochs=50)
    # Sla op in cache
    MODEL_CACHE[site_id] = {
        "status": "ready",
        "trained_at": now.isoformat(),
        "rows_used": len(df_prepared),
        "model": model,
        "scaler": scaler,
        "processor": processor,
        "model_id": "improved_lstm_attention_v1",
    }
    return {
        "site_id": site_id,
        "status": "trained",
        "rows_used": len(df_prepared),
        "model_id": "improved_lstm_attention_v1",
    }


def run_forecast_for_site(site_id: str):
    now = datetime.now(timezone.utc)
    hist_start = now - timedelta(days=60)
    horizon_end = now + timedelta(hours=24)

    measurements = _fetch_measurements(site_id, hist_start, now)
    weather_fc = _fetch_weather_forecast(site_id, now - timedelta(hours=1), horizon_end)

    metadata = None
    pv_fc = _estimate_pv_from_weather(weather_fc, metadata)
    ev_fc = _estimate_ev_baseline(measurements)

    model_id = "fallback_average_profile_v1"

    try:
        if not measurements.empty:
            if site_id not in MODEL_CACHE or "model" not in MODEL_CACHE.get(site_id, {}):
                train_result = train_models_for_site(site_id)
                if "error" in train_result:
                    raise RuntimeError(train_result["error"])

            cache = MODEL_CACHE.get(site_id, {})
            processor = cache.get("processor")
            model = cache.get("model")
            scaler = cache.get("scaler")

            if processor and model and scaler:
                df_local = measurements.rename(columns={
                    "ts_utc": "timestamp",
                    "power_kw": "value"
                })

                df_prepared = processor.prepare_features(df_local)

                if len(df_prepared) >= 96:
                    load_fc = []
                    last_df = df_prepared.copy()

                    for _ in range(96):
                        pred = predict_future(model, scaler, last_df)
                        pred_val = float(pred[0])
                        load_fc.append(pred_val)

                        next_ts = last_df["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)
                        new_row = pd.DataFrame({
                            "timestamp": [next_ts],
                            "value": [pred_val]
                        })

                        new_row = processor.add_time_features(new_row)

                        for lag in processor.lags:
                            new_row[f"lag_{lag}"] = np.nan

                        last_df = pd.concat([last_df, new_row], ignore_index=True)
                        last_df = processor.add_lag_features(last_df)
                        last_df = last_df.dropna().reset_index(drop=True)

                    load_fc = _sanitize_array(np.array(load_fc))
                    lower = np.zeros_like(load_fc)
                    upper = np.zeros_like(load_fc)
                    model_id = cache.get("model_id", "improved_lstm_attention_v1")
                else:
                    load_fc, lower, upper = _fallback_load_forecast(measurements)
            else:
                load_fc, lower, upper = _fallback_load_forecast(measurements)
        else:
            load_fc = np.zeros(96)
            lower = np.zeros(96)
            upper = np.zeros(96)

    except Exception as e:
        print(f"[forecast] fallback used for site {site_id}: {e}")
        load_fc, lower, upper = _fallback_load_forecast(measurements)

    _save_forecast(site_id, load_fc, pv_fc, ev_fc, lower, upper, model_id)

    return {
        "site_id": site_id,
        "status": "forecast_created",
        "model_id": model_id,
        "rows_written": 96,
        "weather_available": not weather_fc.empty,
        "fallback_used": model_id == "fallback_average_profile_v1",
    }


def _save_forecast(site_id: str, load_fc: np.ndarray, pv_fc: np.ndarray, ev_fc: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray, model_id: str) -> None:
    """Schrijf de voorspellingen weg in de database `forecast_predictions_15m`."""
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

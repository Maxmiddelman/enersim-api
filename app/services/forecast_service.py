"""
forecast_service
================

Deze module bevat de forecastlogica voor EnerSim.  In plaats van één
samengestelde voorspelling gebruiken we drie losse modellen voor load,
zonne‑opwek en EV‑laadvermogen.  De uiteindelijke netvoorspelling wordt
vervolgens berekend als:

::

    net_kw = load_kw + ev_kw - pv_kw

Het load‑model is een LSTM dat op historisch verbruik traint en 96
kwartieren vooruit voorspelt.  Het PV‑model is een GradientBoosting
regressor dat gebruik maakt van weersfeatures om de zonopwek te
schatten.  Het EV‑model is een eenvoudig baseline‑model dat per
kwartier het gemiddelde van de afgelopen week gebruikt.  Voor elke
locatie worden de modellen getraind en gecachet in geheugen.  Bij
voorspellen wordt teruggevallen op eenvoudige gemiddelden indien er
onvoldoende data beschikbaar is.

Deze implementatie verwacht dat er Supabase‑tabellen beschikbaar zijn
voor meetdata (``site_measurements_15m``), weersvoorspellingen
(``weather_forecast_hourly``) en output
(``forecast_predictions_15m``).  Het Supabase‑project dient reeds
geconfigureerd te zijn via omgevingsvariabelen; zie ``app/config.py``.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import (
    FORECAST_HORIZON_QUARTERS,
    LSTM_SEQUENCE_LENGTH,
    LOAD_LOOKBACK_DAYS,
    PV_LOOKBACK_DAYS,
)
from app.db import supabase
from app.models.load_lstm import LoadLSTMForecaster
from app.models.pv_model import PVModel
from app.models.ev_model import EVModel
from app.utils.features import (
    add_time_features,
    build_quarter_weather_from_hourly,
    merge_measurements_and_weather,
)


# Cache voor getrainde modellen per site
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
        for col in ["power_kw", "load_kw", "pv_kw", "ev_kw"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _fetch_weather(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    """Haal uurlijkse weersdata op uit Supabase voor een locatie en tijdsinterval."""
    result = (
        supabase.table("weather_forecast_hourly")
        .select("*")
        .eq("site_id", site_id)
        .gte("timestamp", start_ts.isoformat())
        .lte("timestamp", end_ts.isoformat())
        .order("timestamp")
        .execute()
    )
    df = pd.DataFrame(result.data or [])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Converteer numerieke velden; indien veld ontbreekt, blijft lege kolom
        for col in [
            "temperature_2m",
            "cloud_cover",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "wind_speed_10m",
            "precipitation",
            "sunshine_duration",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _compute_load_confidence_bounds(
    measurements: pd.DataFrame, load_fc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Bepaal een eenvoudig betrouwbaarheidsinterval voor de load-voorspelling.

    We gebruiken de standaardafwijking per kwartier van de dag uit de
    historische meetdata om een onder- en bovengrens te berekenen.  Bij te
    weinig data of ontbrekende waarden wordt een vlakke marge gebruikt.

    Parameters
    ----------
    measurements : pd.DataFrame
        Historische meetdata met kolom ``ts_utc`` en ``load_kw``.
    load_fc : np.ndarray
        Voorspelde load‑waarden (lengte 96).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (lower, upper) arrays met lengte 96.
    """
    if measurements.empty or "load_kw" not in measurements.columns:
        lower = load_fc * 0.9
        upper = load_fc * 1.1
        return _sanitize_array(lower), _sanitize_array(upper)

    df = measurements.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["quarter_index"] = df["ts_utc"].dt.hour * 4 + df["ts_utc"].dt.minute // 15
    grouped = df.groupby("quarter_index")["load_kw"]
    std_per_q = grouped.std().reindex(range(96), fill_value=0.0).fillna(0.0).values
    lower = np.maximum(load_fc - std_per_q, 0.0)
    upper = load_fc + std_per_q
    return _sanitize_array(lower), _sanitize_array(upper)


def train_models_for_site(site_id: str) -> Dict[str, Any]:
    """Train load-, pv- en ev-modellen voor een site en sla ze op in de cache."""
    now = datetime.now(timezone.utc)
    load_start = now - timedelta(days=LOAD_LOOKBACK_DAYS)
    pv_start = now - timedelta(days=PV_LOOKBACK_DAYS)
    # Laad meetdata
    measurements = _fetch_measurements(site_id, load_start, now)
    if measurements.empty:
        return {"error": "geen meetdata gevonden", "site_id": site_id}

    # PV-model trainen
    pv_model = PVModel()
    try:
        # Haal historische weersdata (indien beschikbaar)
        weather_hist = _fetch_weather(site_id, pv_start, now)
        if weather_hist.empty:
            # Geen weerdata; PV-model voorspelt nagenoeg nul
            pv_model.model = None  # markeer als leeg
        else:
            # Zet uurlijkse weerdata om naar kwartierwaarden en merge met measurements
            weather_q = build_quarter_weather_from_hourly(weather_hist)
            meas_rename = measurements.rename(columns={"ts_utc": "timestamp"})
            merged = merge_measurements_and_weather(meas_rename, weather_q)
            # Filter alleen rijen met pv_kw > 0 voor training
            if "pv_kw" in merged.columns and not merged["pv_kw"].dropna().empty:
                X = merged[[
                    "temperature_2m",
                    "cloud_cover",
                    "shortwave_radiation",
                    "direct_radiation" if "direct_radiation" in merged.columns else "shortwave_radiation",
                    "diffuse_radiation" if "diffuse_radiation" in merged.columns else "shortwave_radiation",
                    "wind_speed_10m" if "wind_speed_10m" in merged.columns else "cloud_cover",
                    "precipitation" if "precipitation" in merged.columns else "cloud_cover",
                    "sunshine_duration" if "sunshine_duration" in merged.columns else "cloud_cover",
                ]].values
                y = merged["pv_kw"].values
                pv_model.fit(X, y)
            else:
                pv_model.model = None
    except Exception:
        pv_model.model = None

    # EV-model trainen
    ev_model = EVModel(lookback_days=7)
    try:
        meas_ev = measurements.rename(columns={"ts_utc": "timestamp"})
        ev_model.fit(meas_ev)
    except Exception:
        ev_model.baseline = np.zeros(96)

    # Load-model trainen
    load_model = LoadLSTMForecaster(sequence_length=LSTM_SEQUENCE_LENGTH, horizon_steps=FORECAST_HORIZON_QUARTERS)
    try:
        meas_load = measurements.rename(columns={"ts_utc": "timestamp"})
        # Voeg tijdfeatures toe
        meas_load = add_time_features(meas_load, "timestamp")
        # Zorg dat targetkolom aanwezig is; gebruik load_kw als beschikbaar, anders power_kw
        target_col = "load_kw" if "load_kw" in meas_load.columns else "power_kw"
        # Definieer featurekolommen; we kiezen een mix van load zelf en tijdfeatures.
        feature_cols = [
            target_col,
            "hour",
            "day_of_week",
            "sin_qod",
            "cos_qod",
            "sin_dow",
            "cos_dow",
            "sin_doy",
            "cos_doy",
            "is_weekend",
        ]
        # Vul ontbrekende waarden
        meas_load[feature_cols] = meas_load[feature_cols].fillna(0.0)
        load_model.fit(meas_load, feature_cols)
    except Exception as e:
        return {"error": f"fout bij trainen load-model: {e}", "site_id": site_id}

    # Sla in cache
    MODEL_CACHE[site_id] = {
        "trained_at": now.isoformat(),
        "load_model": load_model,
        "pv_model": pv_model,
        "ev_model": ev_model,
    }
    return {
        "site_id": site_id,
        "status": "trained",
        "rows_used": len(measurements),
        "model_ids": {
            "load": "lstm_v1",
            "pv": "gbm_v1" if pv_model.model is not None else "pv_zero",
            "ev": "ev_baseline_v1",
        },
    }


def run_forecast_for_site(site_id: str) -> Dict[str, Any]:
    """Genereer een dag‑ahead forecast voor een locatie."""
    now = datetime.now(timezone.utc)
    # Zorg dat model in cache staat; train indien niet aanwezig
    if site_id not in MODEL_CACHE:
    return {
        "error": "Model not trained. Run /train first before calling /forecast.",
        "site_id": site_id
    }

    cache = MODEL_CACHE.get(site_id, {})
    load_model: Optional[LoadLSTMForecaster] = cache.get("load_model")
    pv_model: Optional[PVModel] = cache.get("pv_model")
    ev_model: Optional[EVModel] = cache.get("ev_model")

    # Bereken tijdsperioden voor input en horizon
    hist_start = now - timedelta(days=max(LOAD_LOOKBACK_DAYS, 7))
    horizon_end = now + timedelta(minutes=15 * FORECAST_HORIZON_QUARTERS)

    # Haal historische metingen voor input
    measurements = _fetch_measurements(site_id, hist_start, now)
    if measurements.empty:
        return {"error": "geen recente meetdata", "site_id": site_id}

    # Load forecast
    try:
        meas_load = measurements.rename(columns={"ts_utc": "timestamp"})
        meas_load = add_time_features(meas_load, "timestamp")
        target_col = "load_kw" if "load_kw" in meas_load.columns else "power_kw"
        feature_cols = [
            target_col,
            "hour",
            "day_of_week",
            "sin_qod",
            "cos_qod",
            "sin_dow",
            "cos_dow",
            "sin_doy",
            "cos_doy",
            "is_weekend",
        ]
        meas_load[feature_cols] = meas_load[feature_cols].fillna(0.0)
        load_fc = load_model.predict(meas_load)
        load_fc = _sanitize_array(load_fc)
    except Exception:
        # fallback: gebruik gemiddeld profiel op basis van laatst beschikbare data
        load_fc, lower, upper = _fallback_load_forecast(measurements)
        pv_fc = np.zeros_like(load_fc)
        ev_fc = np.zeros_like(load_fc)
        _save_forecast(site_id, load_fc, pv_fc, ev_fc, lower, upper, "fallback_average_profile_v1")
        return {
            "site_id": site_id,
            "status": "forecast_created",
            "model_ids": {"load": "fallback_average_profile_v1", "pv": "pv_zero", "ev": "ev_baseline_v1"},
            "rows_written": 96,
        }

    # PV forecast voor komende 24 uur
    pv_fc = np.zeros(FORECAST_HORIZON_QUARTERS)
    try:
        # Haal weersvoorspelling
        weather_fc = _fetch_weather(site_id, now, horizon_end)
        if not weather_fc.empty and pv_model is not None and pv_model.model is not None:
            weather_q = build_quarter_weather_from_hourly(weather_fc)
            # Gebruik dezelfde volgorde van features als bij training
            feature_list = [
                "temperature_2m",
                "cloud_cover",
                "shortwave_radiation",
                "direct_radiation" if "direct_radiation" in weather_q.columns else "shortwave_radiation",
                "diffuse_radiation" if "diffuse_radiation" in weather_q.columns else "shortwave_radiation",
                "wind_speed_10m" if "wind_speed_10m" in weather_q.columns else "cloud_cover",
                "precipitation" if "precipitation" in weather_q.columns else "cloud_cover",
                "sunshine_duration" if "sunshine_duration" in weather_q.columns else "cloud_cover",
            ]
            X_fc = weather_q[feature_list].head(FORECAST_HORIZON_QUARTERS).values
            preds = pv_model.predict(X_fc)
            # Zorg op lengte 96
            pv_fc[: len(preds)] = preds
            pv_fc = np.clip(pv_fc, 0.0, None)
    except Exception:
        pv_fc = np.zeros(FORECAST_HORIZON_QUARTERS)

    # EV forecast (baseline)
    try:
        ev_fc = ev_model.predict()
    except Exception:
        ev_fc = np.zeros(FORECAST_HORIZON_QUARTERS)

    # Bereken betrouwbaarheidsmarges
    lower, upper = _compute_load_confidence_bounds(measurements, load_fc)

    # Sla forecast op
    _save_forecast(site_id, load_fc, pv_fc, ev_fc, lower, upper, "ensemble_v1")
    return {
        "site_id": site_id,
        "status": "forecast_created",
        "model_ids": {
            "load": "lstm_v1",
            "pv": "gbm_v1" if pv_model.model is not None else "pv_zero",
            "ev": "ev_baseline_v1",
        },
        "rows_written": FORECAST_HORIZON_QUARTERS,
    }


def _fallback_load_forecast(measurements: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback load‑voorspelling gebaseerd op gemiddeld kwartierprofiel."""
    if measurements.empty or "power_kw" not in measurements.columns:
        load_fc = np.zeros(FORECAST_HORIZON_QUARTERS)
        return load_fc, load_fc.copy(), load_fc.copy()
    df = measurements.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["quarter_index"] = df["ts_utc"].dt.hour * 4 + df["ts_utc"].dt.minute // 15
    grouped = df.groupby("quarter_index")["power_kw"]
    mean_per_q = grouped.mean()
    std_per_q = grouped.std().fillna(0)
    global_mean = mean_per_q.mean() if not mean_per_q.empty else 0.0
    global_std = std_per_q.mean() if not std_per_q.empty else 0.0
    load_fc = np.zeros(FORECAST_HORIZON_QUARTERS)
    lower = np.zeros(FORECAST_HORIZON_QUARTERS)
    upper = np.zeros(FORECAST_HORIZON_QUARTERS)
    for q in range(FORECAST_HORIZON_QUARTERS):
        m = mean_per_q.get(q, global_mean)
        s = std_per_q.get(q, global_std)
        load_fc[q] = m
        lower[q] = max(m - s, 0.0)
        upper[q] = m + s
    return _sanitize_array(load_fc), _sanitize_array(lower), _sanitize_array(upper)


def _save_forecast(
    site_id: str,
    load_fc: np.ndarray,
    pv_fc: np.ndarray,
    ev_fc: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    model_id: str,
) -> None:
    """Schrijf de voorspellingen weg in Supabase in de tabel
    ``forecast_predictions_15m``."""
    now = datetime.now(timezone.utc)
    # Rond af naar het dichtstbijzijnde kwartier
    start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    rows = []
    for i in range(FORECAST_HORIZON_QUARTERS):
        ts = start + timedelta(minutes=15 * i)
        load_val = _safe_float(load_fc[i])
        pv_val = _safe_float(pv_fc[i])
        ev_val = _safe_float(ev_fc[i])
        net_val = _safe_float(load_val + ev_val - pv_val)
        lower_val = _safe_float(lower[i])
        upper_val = _safe_float(upper[i])
        rows.append(
            {
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
                "metadata": {"source": "enersim-api", "version": model_id},
            }
        )
    supabase.table("forecast_predictions_15m").upsert(rows).execute()

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from app.db import supabase
from app.config import (
    FORECAST_HORIZON_QUARTERS,
    LSTM_SEQUENCE_LENGTH,
    PV_LOOKBACK_DAYS,
    LOAD_LOOKBACK_DAYS,
)
from app.models.pv_model import PVModel
from app.models.load_lstm import LoadLSTMForecaster
from app.utils.features import (
    build_quarter_weather_from_hourly,
    merge_measurements_and_weather,
    add_lag_features,
    add_rolling_features,
)

# In-memory cache voor snelheid
MODEL_CACHE = {}


def _fetch_site(site_id: str):
    result = supabase.table("sites").select("*").eq("id", site_id).single().execute()
    return result.data


def _fetch_measurements(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    result = (
        supabase.table("site_measurements_15m")
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
    return df


def _fetch_weather_history(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    result = (
        supabase.table("weather_history_hourly")
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
    return df


def _fetch_weather_forecast(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
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
    return df


def _build_training_frame(site_id: str):
    now = datetime.now(timezone.utc)
    load_start = now - timedelta(days=LOAD_LOOKBACK_DAYS)
    pv_start = now - timedelta(days=PV_LOOKBACK_DAYS)

    measurements = _fetch_measurements(site_id, load_start, now)
    weather_hist = _fetch_weather_history(site_id, load_start, now)

    if measurements.empty:
        raise ValueError("Geen metingen gevonden voor deze site.")
    if weather_hist.empty:
        raise ValueError("Geen historische weather gevonden voor deze site.")

    weather_q = build_quarter_weather_from_hourly(weather_hist)
    df = merge_measurements_and_weather(measurements, weather_q)

    df = add_lag_features(df, "load_kw", [1, 2, 4, 96, 192, 672])
    df = add_rolling_features(df, "load_kw", [4, 12, 96])

    df = df.ffill().bfill().dropna().reset_index(drop=True)
    return df


def train_models_for_site(site_id: str):
    site = _fetch_site(site_id)
    if not site:
        return {"error": "site not found"}

    df = _build_training_frame(site_id)

    # PV model
    pv_model = PVModel()
    pv_feature_cols = [
        "shortwave_radiation",
        "temperature_2m",
        "cloud_cover",
        "sin_qod",
        "cos_qod",
        "sin_doy",
        "cos_doy",
        "is_weekend",
    ]

    pv_train = df[pd.notnull(df["pv_kw"])].copy()
    pv_model.fit(pv_train[pv_feature_cols], pv_train["pv_kw"])
    pv_pred_train = pv_model.predict(pv_train[pv_feature_cols])

    # Voeg pv voorspelling toe als feature voor load
    df["pv_pred_hist"] = pv_model.predict(df[pv_feature_cols])

    load_feature_cols = [
        "load_kw",
        "pv_pred_hist",
        "ev_kw",
        "temperature_2m",
        "cloud_cover",
        "sin_qod",
        "cos_qod",
        "sin_dow",
        "cos_dow",
        "load_kw_lag_1",
        "load_kw_lag_4",
        "load_kw_lag_96",
        "load_kw_roll_mean_4",
        "load_kw_roll_mean_12",
        "load_kw_roll_mean_96",
    ]

    lstm = LoadLSTMForecaster(
        sequence_length=LSTM_SEQUENCE_LENGTH,
        horizon_steps=FORECAST_HORIZON_QUARTERS,
    )
    lstm.fit(df, load_feature_cols, epochs=8, batch_size=64, lr=0.001)

    MODEL_CACHE[site_id] = {
        "pv_model": pv_model,
        "load_lstm": lstm,
        "pv_feature_cols": pv_feature_cols,
        "load_feature_cols": load_feature_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "site_id": site_id,
        "status": "trained",
        "rows_used": len(df),
        "pv_train_mae": float(np.mean(np.abs(pv_train["pv_kw"].values - pv_pred_train))),
    }


def _ensure_models(site_id: str):
    if site_id not in MODEL_CACHE:
        train_models_for_site(site_id)


def _make_future_feature_frame(site_id: str, recent_df: pd.DataFrame):
    now = datetime.now(timezone.utc)
    horizon_end = now + timedelta(minutes=15 * FORECAST_HORIZON_QUARTERS)

    weather_fc = _fetch_weather_forecast(site_id, now - timedelta(hours=1), horizon_end)
    if weather_fc.empty:
        raise ValueError("Geen weather forecast beschikbaar.")

    weather_q = build_quarter_weather_from_hourly(weather_fc)

    future = weather_q[["timestamp", "temperature_2m", "cloud_cover", "shortwave_radiation"]].copy()
    future = future.sort_values("timestamp").head(FORECAST_HORIZON_QUARTERS).reset_index(drop=True)

    # Tijdfeatures toevoegen via merge helper
    empty_measurements = pd.DataFrame({
        "timestamp": future["timestamp"],
        "load_kw": [recent_df["load_kw"].iloc[-1]] * len(future),
        "pv_kw": [0.0] * len(future),
        "ev_kw": [recent_df["ev_kw"].iloc[-96:].mean() if len(recent_df) >= 96 else 0.0] * len(future),
    })

    merged = merge_measurements_and_weather(empty_measurements, future)
    return merged


def _save_predictions(site_id: str, pred_df: pd.DataFrame):
    rows = pred_df.to_dict(orient="records")
    if rows:
        supabase.table("forecast_predictions_15m").upsert(rows).execute()


def run_forecast_for_site(site_id: str):
    _ensure_models(site_id)

    models = MODEL_CACHE[site_id]
    pv_model = models["pv_model"]
    lstm = models["load_lstm"]
    pv_feature_cols = models["pv_feature_cols"]
    load_feature_cols = models["load_feature_cols"]

    now = datetime.now(timezone.utc)
    hist_start = now - timedelta(days=14)
    recent_df = _fetch_measurements(site_id, hist_start, now)
    weather_hist = _fetch_weather_history(site_id, hist_start, now)

    if recent_df.empty or weather_hist.empty:
        return {"error": "te weinig recente data"}

    weather_q = build_quarter_weather_from_hourly(weather_hist)
    recent_df = merge_measurements_and_weather(recent_df, weather_q)
    recent_df = add_lag_features(recent_df, "load_kw", [1, 2, 4, 96, 192, 672])
    recent_df = add_rolling_features(recent_df, "load_kw", [4, 12, 96])
    recent_df = recent_df.ffill().bfill().dropna().reset_index(drop=True)

    future_df = _make_future_feature_frame(site_id, recent_df)

    # PV forecast
    pv_future = pv_model.predict(future_df[pv_feature_cols])
    future_df["pv_pred_hist"] = pv_future

    # EV baseline = gemiddelde zelfde kwartier van laatste 7 dagen
    recent_ev = recent_df.tail(96 * 7)["ev_kw"].values
    if len(recent_ev) >= 96:
        ev_template = recent_ev[-96:]
    else:
        ev_template = np.zeros(96)

    future_df["ev_kw"] = ev_template[: len(future_df)]

    # Voor load LSTM hebben we een frame nodig met laatste geschiedenis + future feature rows
    combined = pd.concat([recent_df, future_df], ignore_index=True, sort=False)

    # Lags/rolling opnieuw over combined zodat future rows ook features hebben
    combined["pv_pred_hist"] = combined["pv_pred_hist"].ffill().bfill()
    combined = add_lag_features(combined, "load_kw", [1, 2, 4, 96, 192, 672])
    combined = add_rolling_features(combined, "load_kw", [4, 12, 96])

    combined = combined.ffill().bfill()

    load_preds = lstm.predict(combined)

    result = future_df[["timestamp"]].copy()
    result["site_id"] = site_id
    result["pred_load_kw"] = load_preds[: len(result)]
    result["pred_pv_kw"] = pv_future[: len(result)]
    result["pred_ev_kw"] = future_df["ev_kw"].values[: len(result)]
    result["pred_net_kw"] = result["pred_load_kw"] - result["pred_pv_kw"] + result["pred_ev_kw"]

    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["load_model_version"] = "lstm_v1"
    result["pv_model_version"] = "gbr_v1"
    result["ev_model_version"] = "baseline_v1"

    _save_predictions(site_id, result)

    return {
        "site_id": site_id,
        "status": "forecast_created",
        "rows_written": len(result),
        "trained_at": models["trained_at"],
        "sample": result.head(5).to_dict(orient="records"),
    }

import numpy as np
import pandas as pd


def quarter_of_day(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 4 + (ts.dt.minute // 15)


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_col], utc=True)

    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["quarter_of_day"] = quarter_of_day(ts)

    df["sin_qod"] = np.sin(2 * np.pi * df["quarter_of_day"] / 96)
    df["cos_qod"] = np.cos(2 * np.pi * df["quarter_of_day"] / 96)

    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def build_quarter_weather_from_hourly(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Zet hourly weather om naar kwartierwaarden door ieder uur 4x te herhalen.
    """
    if weather_df.empty:
        return weather_df

    rows = []
    for _, row in weather_df.iterrows():
        base_ts = pd.to_datetime(row["timestamp"], utc=True)
        for minute in [0, 15, 30, 45]:
            new_row = row.to_dict()
            new_row["timestamp"] = base_ts + pd.Timedelta(minutes=minute)
            rows.append(new_row)

    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return out


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
    return df


def clean_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.ffill().bfill()


def merge_measurements_and_weather(measurements_df: pd.DataFrame, weather_q_df: pd.DataFrame) -> pd.DataFrame:
    df = measurements_df.copy()
    weather = weather_q_df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)

    merged = pd.merge(df, weather, on="timestamp", how="left")
    merged = clean_and_fill(merged)
    merged = add_time_features(merged, "timestamp")
    return merged

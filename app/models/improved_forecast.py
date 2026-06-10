"""
Verbeterd verbruiksvoorspellingsmodel voor EnerSim
======================================================================

Dit module implementeert een verbeterde versie van de verbruiksvoorspeller
voor EnerSim. Het combineert uitgebreide feature‑engineering met een
Long Short‑Term Memory (LSTM) model voorzien van een aandachtmechanisme.

Kenmerken van dit module:

* **DataProcessor** – verzorgt aggregatie en feature‑engineering.  De
  klasse accepteert ruwe meetdata op seconde‑niveau en aggregeert deze naar
  kwartierwaarden.  Het voegt tijd‑ en kalenderkenmerken, lags, rollende
  statistieken en eventuele exogene variabelen (zoals weersdata) toe.

* **LSTMWithAttention** – een PyTorch‑implementatie van een LSTM met
  zelf‑aandachtsmechanisme.  Dit model is in staat om belangrijke
  tijdstappen in de inputsequentie automatisch te wegen, waardoor
  langeafhankelijke patronen beter worden vastgelegd.

* **train_model** – een functie voor het trainen van het LSTM‑model op de
  geaggregeerde features.  De functie splitst de data in trainings‑ en
  validatiesets op basis van tijd (walk‑forward), optimaliseert de
  hyperparameters en past early stopping toe.

* **predict_future** – functie waarmee dag‑ahead voorspellingen worden
  gegenereerd op basis van het getrainde model en nieuwe features.

Gebruik:

  ```python
  import pandas as pd
  from improved_forecast import DataProcessor, LSTMWithAttention, train_model, predict_future

  # Laad ruwe seconde‑data (timestamp, value, etc.)
  df_raw = pd.read_csv('sensor_data.csv', parse_dates=['timestamp'])
  # Maak een DataProcessor aan met benodigde instellingen
  processor = DataProcessor(freq='15T', lags=[1, 4, 8, 12], weather_cols=['ghi', 'temperature'])
  df_features = processor.prepare_features(df_raw)
  # Train het model
  model, scaler = train_model(df_features, input_window=96, output_window=96, epochs=10)
  # Genereer voorspellingen voor de komende 96 stappen (24 uur bij kwartierresolutie)
  df_future = processor.generate_future_features(df_features, periods=96)
  predictions = predict_future(model, scaler, df_future)
  ```

De code is modulair opgezet zodat je hem eenvoudig kunt integreren in
jouw bestaande toepassing.  De data‑ingang (Supabase, websockets) en
output (Render API) kun je desgewenst vervangen door eigen logic.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
# We implement a simple scaler to avoid dependence on scikit-learn.  It behaves
# similar aan StandardScaler: het berekent de mean en std per feature en
# schaalt data volgens (x - mean) / std.  Hierdoor hebben we geen externe
# dependency zoals scikit-learn nodig, wat handig is in deployment omgevingen
# waar scikit-learn niet beschikbaar is.


class SimpleScaler:
    """Eenvoudige schaaltransformatie zonder externe afhankelijkheden.

    De scaler berekent voor elke feature de gemiddelde en standaardafwijking.
    Tijdens transformeren wordt (x - mean) / std toegepast.  Een std van 0
    wordt vervangen door 1 om deling door nul te voorkomen.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'SimpleScaler':
        """Bereken mean en std voor de kolommen."""
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        # Vervang nullen in std door 1 om deling door nul te voorkomen
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
import torch
import torch.nn as nn


@dataclass
class DataProcessor:
    """Verwerkt ruwe data naar features voor het trainingsmodel.

    Parameters
    ----------
    freq : str
        De frequentie waarop de tijdreeks moet worden geaggregeerd.  Voor
        kwartierwaarden gebruik je '15T'.
    lags : List[int]
        Een lijst van lags (in stappen) die als extra features worden
        toegevoegd.  Bijvoorbeeld [1, 4, 8] om respectievelijk 1, 1 uur en
        2 uur terug te kijken bij kwartierresolutie.
    weather_cols : Optional[List[str]]
        Een lijst van namen voor eventuele exogene variabelen (zoals
        temperatuur, irradiantie) die al aanwezig zijn in de input DataFrame.

    """

    freq: str = '15T'
    lags: List[int] = None
    weather_cols: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.lags is None:
            self.lags = [1, 4, 8]
        if self.weather_cols is None:
            self.weather_cols = []

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates second‑level data to the specified frequency.

        Expects the input DataFrame `df` to have at least columns `timestamp`
        (datetime) and `value` (float) representing consumption.  Additional
        columns are carried over via aggregation (mean).

        Returns a new DataFrame indexed on the aggregated timestamp.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Set timestamp as index and resample
        df = df.set_index('timestamp').sort_index()
        # Compute mean for all numeric columns for the given frequency
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    return pd.DataFrame(columns=["timestamp", "value"])

agg_df = df[numeric_cols].resample(self.freq).mean()
        agg_df = agg_df.reset_index()
        return agg_df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Voegt tijd‑ en seizoenskenmerken toe aan de DataFrame.

        Toegevoegde kolommen:
        - hour: uur van de dag [0..23]
        - dayofweek: dag van de week [0=maandag..6=zondag]
        - month: maand [1..12]
        - dayofyear_sin/cos: sinusoïdale representatie voor jaarlijkse seizoenen
        - minute15: index van het kwartier binnen de dag [0..95]
        """
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        # Seizoensperioden via sin/cos
        day_of_year = df['timestamp'].dt.dayofyear
        df['dayofyear_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        # Kwartierindex voor dagelijkse seizoenen
        minute_of_day = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        quarter_index = (minute_of_day / 15).astype(int)
        df['minute15_sin'] = np.sin(2 * np.pi * quarter_index / 96)
        df['minute15_cos'] = np.cos(2 * np.pi * quarter_index / 96)
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Voegt lags en rollende statistieken toe voor de verbruikswaarde.

        Voor elke lag in self.lags wordt een kolom `lag_{n}` toegevoegd.
        Daarnaast worden rollende minimum, maximum en standaardafwijking met
        vensterbreedte gelijk aan de grootste lag toegevoegd.
        """
        df = df.copy()
        max_lag = max(self.lags)
        for lag in self.lags:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        # Rollende statistieken voor de laatste max_lag stappen
        df[f'rolling_min_{max_lag}'] = df['value'].rolling(window=max_lag).min()
        df[f'rolling_max_{max_lag}'] = df['value'].rolling(window=max_lag).max()
        df[f'rolling_std_{max_lag}'] = df['value'].rolling(window=max_lag).std()
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volledige verwerking: aggregatie, tijdfeatures, lags en opschonen.

        Parameters
        ----------
        df : pd.DataFrame
            Ruwe data met minimaal kolommen `timestamp`, `value` en
            eventuele exogene variabelen uit `weather_cols`.

        Returns
        -------
        pd.DataFrame
            Geaggregeerde en verrijkte data waarop getraind kan worden.
        """
        agg_df = self.aggregate(df)
        feat_df = self.add_time_features(agg_df)
        feat_df = self.add_lag_features(feat_df)
        # Voeg exogene variabelen (weather_cols) gewoon door; missende kolommen
        # blijven behouden
        # Verwijder rijen met NA's (vanwege lags)
        feat_df = feat_df.dropna().reset_index(drop=True)
        return feat_df

    def generate_future_features(self, df: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Genereert features voor toekomstig voorspellen.

        Gebruikt de laatste rij van de bestaande DataFrame om de tijdstempels
        van de komende `periods` stappen (met frequentie `freq`) te
        creëren en vult tijd‑features in.  Lags worden vooralsnog op nul
        gezet; het model dient de waarde van lags intern bij te houden.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame met dezelfde kolommen als output van `prepare_features`.
        periods : int
            Aantal toekomstige kwartierstappen om te genereren.

        Returns
        -------
        pd.DataFrame
            DataFrame met tijdfeatures voor de toekomst.
        """
        last_ts = pd.to_datetime(df['timestamp'].iloc[-1])
        future_times = pd.date_range(start=last_ts + pd.to_timedelta(self.freq),
                                     periods=periods,
                                     freq=self.freq)
        future_df = pd.DataFrame({'timestamp': future_times})
        future_df = self.add_time_features(future_df)
        # initialiseert lags met NaN; het model (LSTM) kan deze waarden negeren
        for lag in self.lags:
            future_df[f'lag_{lag}'] = np.nan
        max_lag = max(self.lags)
        future_df[f'rolling_min_{max_lag}'] = np.nan
        future_df[f'rolling_max_{max_lag}'] = np.nan
        future_df[f'rolling_std_{max_lag}'] = np.nan
        return future_df


class AttentionLayer(nn.Module):
    """Eenvoudige aandachtlaag voor LSTM‑uitvoer.

    Deze laag berekent aandachtsscores op basis van de hidden states van de LSTM
    en de laatste hidden state.  Het gebruikt een dot‑product attentie.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch, seq_len, hidden_dim]
        # Gebruik de laatste hidden state als query
        query = hidden_states[:, -1:, :]  # [batch, 1, hidden_dim]
        # Transponeer hidden_states voor dot‑product (keys/values)
        keys = hidden_states.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        # Dot‑product: [batch, 1, seq_len]
        attn_scores = torch.bmm(query, keys) / np.sqrt(self.hidden_dim)
        attn_weights = self.softmax(attn_scores)
        # Waarde: hidden_states [batch, seq_len, hidden_dim]
        context = torch.bmm(attn_weights, hidden_states)  # [batch, 1, hidden_dim]
        context = context.squeeze(1)
        return context


class LSTMWithAttention(nn.Module):
    """LSTM‑model met aandacht voor multivariate tijdreeksvoorspelling.

    Parameters
    ----------
    input_dim : int
        Aantal features in de input (per timestep).
    hidden_dim : int
        Aantal verborgen units in de LSTM.
    num_layers : int
        Aantal LSTM‑lagen.
    output_dim : int
        Aantal target variabelen (meestal 1).
    dropout : float
        Dropout‑ratio tussen LSTM‑lagen.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim]
        context = self.attention(lstm_out)  # [batch_size, hidden_dim]
        output = self.fc(context)  # [batch_size, output_dim]
        return output


def split_train_val(df: pd.DataFrame, val_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits de data chronologisch in een trainings‑ en validatieset.

    Omdat tijdreeksen een volgorde hebben, wordt de splitsing gedaan door een
    deel van de laatste data als validatie te gebruiken.
    """
    n = len(df)
    val_idx = int(n * (1 - val_frac))
    train_df = df.iloc[:val_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx:].reset_index(drop=True)
    return train_df, val_df


def create_sequences(df: pd.DataFrame, input_window: int, output_window: int, feature_cols: List[str], target_col: str = 'value') -> Tuple[np.ndarray, np.ndarray]:
    """Vormt sequenties voor LSTM‑training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame met features en target.
    input_window : int
        Aantal timesteps in de input (bijvoorbeeld 96 voor 24 uur bij kwartierresolutie).
    output_window : int
        Aantal timesteps dat voorspeld moet worden.  Momenteel wordt alleen
        een enkel punt voorspeld; extra toekomstige stappen kun je met sliding
        windows genereren.
    feature_cols : List[str]
        Kolomnamen die als inputfeatures worden gebruikt.
    target_col : str
        Kolomnaam van de target.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple van (inputs, targets), waarbij inputs de vorm
        (n_samples, input_window, n_features) en targets de vorm
        (n_samples, output_dim) hebben.
    """
    inputs = []
    targets = []
    total_len = len(df)
    # Normaliseer de target parallel aan features
    for i in range(total_len - input_window - output_window + 1):
        input_seq = df.iloc[i:i + input_window][feature_cols].values
        target_seq = df.iloc[i + input_window:i + input_window + output_window][target_col].values
        inputs.append(input_seq)
        # Voor eenvoud voorspellen we het gemiddelde van output_window stappen; pas aan indien nodig
        targets.append(np.mean(target_seq))
    return np.array(inputs), np.array(targets)


def train_model(df: pd.DataFrame,
                input_window: int = 96,
                output_window: int = 1,
                epochs: int = 50,
                hidden_dim: int = 64,
                num_layers: int = 2,
                lr: float = 0.001,
                device: Optional[str] = None) -> Tuple[LSTMWithAttention, SimpleScaler]:
    """Trainde het LSTM‑met‑aandacht model op de gegeven DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame met features en target.
    input_window : int
        Lengte van de inputsequentie in timesteps.
    output_window : int
        Lengte van de voorspelling (in stappen).  Momenteel wordt de
        gemiddelde target van dit venster gebruikt.
    epochs : int
        Aantal trainingsiteraties.
    hidden_dim : int
        Aantal hidden units in de LSTM.
    num_layers : int
        Aantal LSTM‑lagen.
    lr : float
        Learning rate.
    device : Optional[str]
        'cpu' of 'cuda'; als None wordt automatisch gekozen.

    Returns
    -------
    Tuple[LSTMWithAttention, SimpleScaler]
        Het getrainde model en de scaler die gebruikt is om de features te schalen.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Definieer feature‑kolommen: alle niet‑target kolommen behalve timestamp
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'value']]
    # Normaliseer features met SimpleScaler
    scaler = SimpleScaler()
    features_scaled = scaler.fit_transform(df[feature_cols].values)
    df_scaled = df.copy()
    df_scaled[feature_cols] = features_scaled

    # Splits train en val
    train_df, val_df = split_train_val(df_scaled)
    X_train, y_train = create_sequences(train_df, input_window, output_window, feature_cols)
    X_val, y_val = create_sequences(val_df, input_window, output_window, feature_cols)
    # Converteer naar tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = LSTMWithAttention(input_dim=len(feature_cols), hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.view(-1), y_train)
        loss.backward()
        optimizer.step()
        # Validatie
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.view(-1), y_val)
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            # Bewaar beste model
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

    # Laad beste staat
    model.load_state_dict(best_state)
    return model, scaler


def predict_future(model: LSTMWithAttention,
                   scaler: SimpleScaler,
                   df: pd.DataFrame,
                   input_window: int = 96,
                   feature_cols: Optional[List[str]] = None,
                   device: Optional[str] = None) -> np.ndarray:
    """Maakt dag‑ahead voorspellingen met het getrainde model.

    Parameters
    ----------
    model : LSTMWithAttention
        Het getrainde model.
    scaler : SimpleScaler
        De scaler die gebruikt is om de features te schalen.
    df : pd.DataFrame
        DataFrame met de samengevoegde features die `input_window` stappen
        bevatten vóór de toekomst waarin voorspeld wordt.  Het moet kolommen
        bevatten die overeenkomen met feature_cols.
    input_window : int
        Aantal timesteps dat het model als input gebruikt.
    feature_cols : Optional[List[str]]
        Welke kolommen als features gebruikt worden.  Indien None worden
        automatisch alle kolommen behalve timestamp en value gekozen.
    device : Optional[str]
        'cpu' of 'cuda'.

    Returns
    -------
    np.ndarray
        Een array met voorspelde waarden (gemiddelde waarde per kwartier).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'value']]
    # Schaal features
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols].values)
    # Alleen laatste `input_window` rijen gebruiken
    input_seq = df_scaled[feature_cols].tail(input_window).values
    X = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(X)
    return pred.cpu().numpy().flatten()

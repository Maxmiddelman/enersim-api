"""
EVModel
=======

Deze module implementeert een eenvoudig EV‑laadvermogensmodel.  In veel
gebouwsinstallaties blijkt het laadpatroon van elektrische voertuigen vrij
regelmatig te zijn.  Voor een eerste versie van EnerSim is een
historisch gemiddelde per kwartier vaak voldoende nauwkeurig.  Het model
leert een baseline op basis van de laatste dagen en gebruikt deze
baseline voor dag‑ahead voorspellingen.

Gebruik:

.. code-block:: python

    from app.models.ev_model import EVModel

    ev_model = EVModel(lookback_days=7)
    ev_model.fit(df_measurements)
    preds = ev_model.predict()

Het ``fit``-proces verwacht een DataFrame met ten minste een kolom
``ev_kw`` en een kolom ``timestamp``.  Bij ``predict`` wordt een
array met 96 kwartierwaarden voor de komende 24 uur geretourneerd.

Het model slaat de baseline intern op.  Indien er niet genoeg data
beschikbaar is, wordt een vlakke nulvoorspelling geretourneerd.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd


class EVModel:
    """Eenvoudig baseline‑model voor EV‑laadvermogen.

    Parameters
    ----------
    lookback_days : int, optional
        Aantal dagen om terug te kijken voor het berekenen van de baseline.
        Standaard is 7 dagen.
    """

    def __init__(self, lookback_days: int = 7) -> None:
        self.lookback_days = lookback_days
        self.baseline: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> None:
        """Bereken de gemiddelde EV‑vermogenscurve voor iedere kwartier van de dag.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame met kolommen ``timestamp`` en ``ev_kw``.
        """
        if df.empty or "ev_kw" not in df.columns:
            self.baseline = np.zeros(96)
            return

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # beperk tot laatste lookback_days dagen
        max_ts = df["timestamp"].max()
        min_ts = max_ts - timedelta(days=self.lookback_days)
        df = df[df["timestamp"] >= min_ts]
        # Compute quarter index (0–95)
        quarter_index = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
        df["quarter_index"] = quarter_index
        grouped = df.groupby("quarter_index")["ev_kw"]
        mean_per_q = grouped.mean().reindex(range(96), fill_value=0.0)
        baseline = mean_per_q.fillna(0.0).values.astype(float)
        self.baseline = baseline

    def predict(self) -> np.ndarray:
        """Geef een voorspelling van het EV‑laadvermogen voor de komende 96 kwartieren.

        Returns
        -------
        np.ndarray
            Array van lengte 96 met voorspelde EV‑vermogens (kW).
        """
        if self.baseline is None:
            return np.zeros(96)
        return self.baseline.copy()
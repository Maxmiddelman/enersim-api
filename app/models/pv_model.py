import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class PVModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

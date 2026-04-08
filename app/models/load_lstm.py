import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LoadLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_steps: int = 96):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


class LoadLSTMForecaster:
    def __init__(self, sequence_length=192, horizon_steps=96):
        self.sequence_length = sequence_length
        self.horizon_steps = horizon_steps
        self.model = None
        self.feature_cols = None
        self.mean_ = None
        self.std_ = None

    def _make_sequences(self, df, feature_cols, target_col="load_kw"):
        X_list = []
        y_list = []

        values = df[feature_cols].values.astype(np.float32)
        target = df[target_col].values.astype(np.float32)

        max_i = len(df) - self.sequence_length - self.horizon_steps + 1
        for i in range(max_i):
            X_seq = values[i:i + self.sequence_length]
            y_seq = target[i + self.sequence_length:i + self.sequence_length + self.horizon_steps]
            X_list.append(X_seq)
            y_list.append(y_seq)

        return np.array(X_list), np.array(y_list)

    def fit(self, df, feature_cols, epochs=8, batch_size=64, lr=0.001):
        self.feature_cols = feature_cols

        feat_values = df[feature_cols].values.astype(np.float32)
        self.mean_ = feat_values.mean(axis=0)
        self.std_ = feat_values.std(axis=0) + 1e-6

        df_train = df.copy()
        df_train[feature_cols] = (feat_values - self.mean_) / self.std_

        X, y = self._make_sequences(df_train, feature_cols)
        if len(X) == 0:
            raise ValueError("Niet genoeg data om LSTM-sequenties te maken.")

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        self.model = LoadLSTM(input_size=len(feature_cols), output_steps=self.horizon_steps)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(X_train.size(0))
            total_loss = 0.0

            for i in range(0, X_train.size(0), batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X_train[idx]
                batch_y = y_train[idx]

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val)
                val_loss = criterion(val_preds, y_val).item()
            self.model.train()

            print(f"Epoch {epoch+1}/{epochs} train_loss={total_loss / X_train.size(0):.4f} val_loss={val_loss:.4f}")

    def predict(self, recent_df):
        if self.model is None:
            raise ValueError("LSTM model is nog niet getraind.")

        seq = recent_df[self.feature_cols].tail(self.sequence_length).values.astype(np.float32)
        seq = (seq - self.mean_) / self.std_
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(x).numpy().flatten()

        return preds

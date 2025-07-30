import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .ingest import get_price
from .features import build_features


class LSTMClassifier(nn.Module):
    """Simple LSTM-based classifier for stock direction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        out = output[:, -1, :]
        return self.fc(out)


def create_sequences(df: pd.DataFrame, window: int = 30):
    """Convert feature dataframe into sequences and labels."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'

    returns = df[price_col].pct_change().shift(-1)
    y = (returns > 0).astype(int)

    feature_df = df.drop(columns=[price_col], errors='ignore').fillna(0)
    X, Y = [], []
    for i in range(len(df) - window - 1):
        seq = feature_df.iloc[i : i + window].values
        label = y.iloc[i + window]
        X.append(seq)
        Y.append(label)
    return np.array(X), np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--window', type=int, default=30)
    args = ap.parse_args()

    price = get_price(args.ticker, args.start, args.end)
    if price.empty or len(price) < args.window + 10:
        raise ValueError('Not enough data for training')

    feat = build_features(price, [])
    X, y = create_sequences(feat, window=args.window)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    model = LSTMClassifier(input_dim=X.shape[2])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch+1}/{args.epochs} - loss: {np.mean(losses):.4f}')

    model_path = f'{args.ticker}_deep_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved as {model_path}')


if __name__ == '__main__':
    main()

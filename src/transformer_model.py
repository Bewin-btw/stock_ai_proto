import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report

from .ingest import get_price
from .features import build_features


class TransformerClassifier(nn.Module):
    """Simple Transformer-based classifier."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        enc = self.encoder(x)
        out = enc[:, -1, :]
        return self.fc(out)


def create_sequences(df: pd.DataFrame, window: int = 30):
    """Convert features into sequences for Transformer."""
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
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=3,
                    help='Early stopping patience')
    args = ap.parse_args()

    price = get_price(args.ticker, args.start, args.end)
    if price.empty or len(price) < args.window + 10:
        raise ValueError('Not enough data for training')

    feat = build_features(price, [])
    X, y = create_sequences(feat, window=args.window)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    split = int(len(X_tensor) * 0.8)
    train_ds = TensorDataset(X_tensor[:split], y_tensor[:split])
    val_ds = TensorDataset(X_tensor[split:], y_tensor[split:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = TransformerClassifier(
        input_dim=X.shape[2],
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    patience = args.patience
    counter = 0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                vloss = criterion(pred, yb)
                val_losses.append(vloss.item())
        val_loss = np.mean(val_losses)
        print(
            f'Epoch {epoch+1}/{args.epochs} - '
            f'train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}'
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on the validation split
    model.eval()
    with torch.no_grad():
        xb = X_tensor[split:]
        y_true = y_tensor[split:].numpy().flatten()
        preds = model(xb)
        probs = torch.sigmoid(preds).numpy().flatten()
        pred_labels = (probs > 0.5).astype(int)
        auc = roc_auc_score(y_true, probs)
        print(f'Validation AUC = {auc:.4f}')
        print("\nValidation Report:")
        print(classification_report(y_true, pred_labels))

    model_path = f'{args.ticker}_transformer_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved as {model_path}')


if __name__ == '__main__':
    main()


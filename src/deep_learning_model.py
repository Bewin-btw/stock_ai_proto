import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report

from .ingest import get_price
from .features import build_features


class LSTMClassifier(nn.Module):
    """LSTM classifier with dropout for better generalization."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        out = output[:, -1, :]
        out = self.dropout(out)
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
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=3,
                    help='Early stopping patience')
    ap.add_argument('--device', default=None,
                    help="Force training on 'cpu' or 'cuda'")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

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

    model = LSTMClassifier(
        input_dim=X.shape[2],
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
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
            xb = xb.to(device)
            yb = yb.to(device)
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
                xb = xb.to(device)
                yb = yb.to(device)
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
        xb = X_tensor[split:].to(device)
        y_true = y_tensor[split:].cpu().numpy().flatten()
        preds = model(xb)
        probs = torch.sigmoid(preds).cpu().numpy().flatten()
        pred_labels = (probs > 0.5).astype(int)
        auc = roc_auc_score(y_true, probs)
        print(f'Validation AUC = {auc:.4f}')
        print("\nValidation Report:")
        print(classification_report(y_true, pred_labels))

    model_path = f'{args.ticker}_deep_model.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'input_dim': X.shape[2],
            'hidden_dim': args.hidden,
            'num_layers': args.layers,
            'dropout': args.dropout,
            'window': args.window,
            'device': device.type,
        },
        model_path,
    )
    print(f'Model saved as {model_path}')


if __name__ == '__main__':
    main()

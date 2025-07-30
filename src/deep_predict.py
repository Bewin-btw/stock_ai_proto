import argparse
import torch
import numpy as np

from .ingest import get_price
from .features import build_features
from .deep_learning_model import LSTMClassifier, create_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--window', type=int, default=None,
                    help='Sequence window; if omitted uses value from model')
    ap.add_argument('--model', default=None,
                    help='Path to saved model (default <ticker>_deep_model.pt)')
    args = ap.parse_args()

    model_path = args.model or f'{args.ticker}_deep_model.pt'
    checkpoint = torch.load(model_path)
    chk_window = args.window or checkpoint.get('window', 30)

    price = get_price(args.ticker, args.start, args.end)
    if price.empty or len(price) < chk_window + 1:
        raise ValueError('Not enough data for prediction')

    feat = build_features(price, [])
    # create sequences using window from checkpoint
    X, _ = create_sequences(feat, window=chk_window)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    hidden = checkpoint.get('hidden_dim', 128)
    layers = checkpoint.get('num_layers', 3)
    dropout = checkpoint.get('dropout', 0.2)
    model = LSTMClassifier(
        input_dim=checkpoint.get('input_dim', X.shape[2]),
        hidden_dim=hidden,
        num_layers=layers,
        dropout=dropout,
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pred = model(X_tensor[-1:])
        prob = torch.sigmoid(pred).item()

    action = 'BUY' if prob > 0.6 else 'HOLD'
    print({'ticker': args.ticker, 'prob_up': prob, 'action': action})


if __name__ == '__main__':
    main()

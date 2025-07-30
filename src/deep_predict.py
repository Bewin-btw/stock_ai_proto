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
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--model', default=None,
                    help='Path to saved model (default <ticker>_deep_model.pt)')
    args = ap.parse_args()

    price = get_price(args.ticker, args.start, args.end)
    if price.empty or len(price) < args.window + 1:
        raise ValueError('Not enough data for prediction')

    feat = build_features(price, [])
    X, _ = create_sequences(feat, window=args.window)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model_path = args.model or f'{args.ticker}_deep_model.pt'
    model = LSTMClassifier(input_dim=X.shape[2])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        pred = model(X_tensor[-1:])
        prob = torch.sigmoid(pred).item()

    action = 'BUY' if prob > 0.6 else 'HOLD'
    print({'ticker': args.ticker, 'prob_up': prob, 'action': action})


if __name__ == '__main__':
    main()

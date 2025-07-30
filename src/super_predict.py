import argparse
import datetime as dt
import joblib

from .ingest import get_price, get_news
from .market_enhanced_model import build_market_enhanced_features
from .fundamentals import add_fundamental_features


ACTIONS = {
    0.7: 'STRONG_BUY',
    0.6: 'BUY',
    0.4: 'HOLD',
    0.3: 'SELL',
    0.0: 'STRONG_SELL',
}


def choose_action(prob: float) -> str:
    for threshold, action in sorted(ACTIONS.items(), reverse=True):
        if prob >= threshold:
            return action
    return 'HOLD'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    args = ap.parse_args()

    today = dt.date.today()
    price = get_price(args.ticker, str(today - dt.timedelta(days=200)), str(today))
    if price.empty:
        print('Error: no price data')
        return

    news = get_news(args.ticker, from_dt=str(today - dt.timedelta(days=7)))
    feat = build_market_enhanced_features(price, str(today - dt.timedelta(days=200)), str(today))
    feat = add_fundamental_features(feat, args.ticker)
    feat = feat.iloc[-1:]

    drop_cols = [c for c in ['date', 'Adj Close'] if c in feat.columns]
    X = feat.drop(columns=drop_cols) if drop_cols else feat

    model_name = f'{args.ticker}_super_model.pkl'
    try:
        model = joblib.load(model_name)
    except FileNotFoundError:
        print(f'Model {model_name} not found. Train it first.')
        return

    prob = model.predict_proba(X)[0][1]
    action = choose_action(prob)
    print({'ticker': args.ticker, 'probability_up': prob, 'action': action})


if __name__ == '__main__':
    main()

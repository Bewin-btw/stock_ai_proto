import argparse, joblib, datetime as dt
from .ingest import get_price, get_news
from .features import build_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    args = ap.parse_args()
    today = dt.date.today()
    price = get_price(args.ticker, start=str(today - dt.timedelta(days=200)), end=str(today))
    
    # Check if data was successfully downloaded
    if price.empty:
        print(f"Error: No data found for ticker '{args.ticker}'.")
        print("This could be because:")
        print("1. The ticker symbol is incorrect")
        print("2. The stock has been delisted")
        print("3. There's no data for the specified date range")
        print("4. Network connectivity issues")
        return
    
    news = get_news(args.ticker, from_dt=str(today - dt.timedelta(days=7)))
    feat = build_features(price, news).iloc[-1:]
    
    # Check if features were successfully built
    if feat.empty:
        print(f"Error: Could not build features for ticker '{args.ticker}'.")
        return
    
    # Check if model exists
    model_path = f'{args.ticker}_model.pkl'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print(f"Please train a model first using: python3 -m src.train --ticker {args.ticker} --start 2020-01-01 --end 2025-01-01")
        return
    
    # Remove any non-feature columns if they exist, but don't assume specific columns
    columns_to_drop = []
    if 'date' in feat.columns:
        columns_to_drop.append('date')
    if 'Adj Close' in feat.columns:
        columns_to_drop.append('Adj Close')
    
    if columns_to_drop:
        feat_clean = feat.drop(columns=columns_to_drop)
    else:
        feat_clean = feat
    
    prob = model.predict_proba(feat_clean)[0][1]
    action = 'BUY' if prob > 0.6 else 'HOLD'
    print({'ticker': args.ticker, 'prob_up': prob, 'action': action})

if __name__ == '__main__':
    main()

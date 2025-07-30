import argparse, joblib, datetime as dt
from .ingest import get_price, get_news
from .features import build_features
from .simple_improved import add_simple_features
from .fundamentals import add_fundamental_features

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
    feat = build_features(price, news)
    
    # Add simple features
    feat = add_simple_features(feat)
    
    # Add fundamental features
    feat = add_fundamental_features(feat, args.ticker)
    
    # Get latest data point
    feat = feat.iloc[-1:]
    
    # Check if features were successfully built
    if feat.empty:
        print(f"Error: Could not build features for ticker '{args.ticker}'.")
        return
    
    # Check if model exists
    model_path = f'{args.ticker}_improved_model.pkl'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print(f"Please train a model first using: python3 -m src.simple_improved --ticker {args.ticker} --start 2020-01-01 --end 2025-01-01")
        return
    
    # Remove any non-feature columns if they exist
    columns_to_drop = []
    if 'date' in feat.columns:
        columns_to_drop.append('date')
    if 'Adj Close' in feat.columns:
        columns_to_drop.append('Adj Close')
    
    if columns_to_drop:
        feat_clean = feat.drop(columns=columns_to_drop)
    else:
        feat_clean = feat
    
    # Make prediction
    prob = model.predict_proba(feat_clean)[0][1]
    
    # Determine action based on probability
    if prob > 0.7:
        action = 'STRONG_BUY'
    elif prob > 0.6:
        action = 'BUY'
    elif prob > 0.4:
        action = 'HOLD'
    elif prob > 0.3:
        action = 'SELL'
    else:
        action = 'STRONG_SELL'
    
    # Get confidence level
    if prob > 0.8 or prob < 0.2:
        confidence = 'HIGH'
    elif prob > 0.6 or prob < 0.4:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    result = {
        'ticker': args.ticker,
        'probability_up': prob,
        'action': action,
        'confidence': confidence,
        'timestamp': str(dt.datetime.now())
    }
    
    print(result)

if __name__ == '__main__':
    main() 
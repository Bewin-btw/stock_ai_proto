import argparse, joblib, datetime as dt
import yfinance as yf
import pandas as pd
from .ingest import get_price, get_news
from .features import build_features
from .simple_improved import add_simple_features
from .fundamentals import add_fundamental_features

def get_current_market_data():
    """Получение текущих рыночных данных."""
    market_data = {}
    
    # Get SPY (S&P 500) for market context
    try:
        spy_data = yf.download('SPY', start=str(dt.date.today() - dt.timedelta(days=30)), 
                              end=str(dt.date.today()), auto_adjust=True)
        if not spy_data.empty:
            market_data['SPY'] = spy_data
    except:
        pass
    
    # Get QQQ (NASDAQ) for tech context
    try:
        qqq_data = yf.download('QQQ', start=str(dt.date.today() - dt.timedelta(days=30)), 
                              end=str(dt.date.today()), auto_adjust=True)
        if not qqq_data.empty:
            market_data['QQQ'] = qqq_data
    except:
        pass
    
    return market_data

def add_market_features(price_df, market_data):
    """Добавление простых рыночных признаков."""
    features = pd.DataFrame(index=price_df.index)
    
    # Get stock price column
    if isinstance(price_df.columns, pd.MultiIndex):
        price_col = ('Close', price_df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    stock_close = price_df[price_col]
    
    # SPY features
    if 'SPY' in market_data:
        spy_close = market_data['SPY']['Close']
        features['spy_momentum'] = spy_close.pct_change(5)
        features['spy_trend'] = (spy_close > spy_close.rolling(50).mean()).astype(int)
        
        # Ensure we're working with Series
        stock_pct = stock_close.pct_change(5)
        spy_pct = spy_close.pct_change(5)
        if isinstance(stock_pct, pd.DataFrame):
            stock_pct = stock_pct.iloc[:, 0]
        if isinstance(spy_pct, pd.DataFrame):
            spy_pct = spy_pct.iloc[:, 0]
        features['vs_spy_performance'] = stock_pct - spy_pct
    
    # QQQ features
    if 'QQQ' in market_data:
        qqq_close = market_data['QQQ']['Close']
        features['qqq_momentum'] = qqq_close.pct_change(5)
        features['tech_trend'] = (qqq_close > qqq_close.rolling(50).mean()).astype(int)
        
        # Ensure we're working with Series
        stock_pct = stock_close.pct_change(5)
        qqq_pct = qqq_close.pct_change(5)
        if isinstance(stock_pct, pd.DataFrame):
            stock_pct = stock_pct.iloc[:, 0]
        if isinstance(qqq_pct, pd.DataFrame):
            qqq_pct = qqq_pct.iloc[:, 0]
        features['vs_qqq_performance'] = stock_pct - qqq_pct
    
    # Fill missing values
    features = features.fillna(0)
    
    return features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    args = ap.parse_args()
    
    today = dt.date.today()
    price = get_price(args.ticker, start=str(today - dt.timedelta(days=200)), end=str(today))
    
    # Check if data was successfully downloaded
    if price.empty:
        print(f"Error: No data found for ticker '{args.ticker}'.")
        return
    
    # Get market data
    market_data = get_current_market_data()
    
    news = get_news(args.ticker, from_dt=str(today - dt.timedelta(days=7)))
    feat = build_features(price, news)
    
    # Add simple features
    feat = add_simple_features(feat)
    
    # Add fundamental features
    feat = add_fundamental_features(feat, args.ticker)
    
    # Add market features
    market_features = add_market_features(price, market_data)
    feat = pd.concat([feat, market_features], axis=1)
    
    # Get latest data point
    feat = feat.iloc[-1:]
    
    # Check if features were successfully built
    if feat.empty:
        print(f"Error: Could not build features for ticker '{args.ticker}'.")
        return
    
    # Check if model exists
    model_path = f'{args.ticker}_market_simple_model.pkl'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print(f"Please train a model first using: python3 -m src.simple_market_model --ticker {args.ticker} --start 2020-01-01 --end 2025-01-01")
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
    
    # Market context
    market_context = "NEUTRAL"
    if 'SPY' in market_data and 'QQQ' in market_data:
        try:
            spy_trend = market_data['SPY']['Close'].iloc[-1] > market_data['SPY']['Close'].rolling(20).mean().iloc[-1]
            qqq_trend = market_data['QQQ']['Close'].iloc[-1] > market_data['QQQ']['Close'].rolling(20).mean().iloc[-1]
            
            if spy_trend and qqq_trend:
                market_context = "BULLISH"
            elif not spy_trend and not qqq_trend:
                market_context = "BEARISH"
        except:
            market_context = "NEUTRAL"
    
    result = {
        'ticker': args.ticker,
        'probability_up': prob,
        'action': action,
        'confidence': confidence,
        'market_context': market_context,
        'timestamp': str(dt.datetime.now())
    }
    
    print(result)

if __name__ == '__main__':
    main() 
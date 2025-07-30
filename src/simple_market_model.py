import argparse, joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd, numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from .ingest import get_price
from .features import build_features
from .simple_improved import add_simple_features, create_simple_labels
from .fundamentals import add_fundamental_features

def get_simple_market_data(start_date, end_date):
    """Получение простых рыночных данных."""
    market_data = {}
    
    # Get SPY (S&P 500) for market context
    try:
        spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
        if not spy_data.empty:
            market_data['SPY'] = spy_data
            print("✓ Downloaded SPY (S&P 500)")
    except:
        print("✗ Failed to download SPY")
    
    # Get QQQ (NASDAQ) for tech context
    try:
        qqq_data = yf.download('QQQ', start=start_date, end=end_date, auto_adjust=True)
        if not qqq_data.empty:
            market_data['QQQ'] = qqq_data
            print("✓ Downloaded QQQ (NASDAQ)")
    except:
        print("✗ Failed to download QQQ")
    
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
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    # Загрузка данных
    price = get_price(args.ticker, args.start, args.end)
    
    if price.empty:
        print(f"Error: No data found for ticker '{args.ticker}'.")
        return
    
    if len(price) < 100:
        print(f"Error: Insufficient data for ticker '{args.ticker}'.")
        return
    
    # Получение рыночных данных
    market_data = get_simple_market_data(args.start, args.end)
    
    # Построение базовых признаков
    feat = build_features(price, [])
    
    # Добавление простых признаков
    feat = add_simple_features(feat)
    
    # Добавление фундаментальных признаков
    feat = add_fundamental_features(feat, args.ticker)
    
    # Добавление рыночных признаков
    market_features = add_market_features(price, market_data)
    feat = pd.concat([feat, market_features], axis=1)
    
    print(f"✓ Total features: {len(feat.columns)}")
    
    # Создание меток
    y = create_simple_labels(price)
    
    # Remove non-feature columns
    columns_to_drop = []
    if 'date' in feat.columns:
        columns_to_drop.append('date')
    if 'Adj Close' in feat.columns:
        columns_to_drop.append('Adj Close')
    
    if columns_to_drop:
        X = feat.drop(columns=columns_to_drop)
    else:
        X = feat
    
    X, y = X.iloc[:-1], y.iloc[:-1]
    
    # Validation
    if X.empty or y.empty:
        print(f"Error: No valid data after processing for ticker '{args.ticker}'.")
        return
    
    # Check class balance
    positive_ratio = y.mean()
    print(f"Positive class ratio: {positive_ratio:.3f}")
    
    # Train/test split
    train = X.index < '2024-01-01'
    train_data = X[train]
    train_labels = y[train]
    
    if len(train_data) < 50:
        print(f"Error: Insufficient training data for ticker '{args.ticker}'.")
        return
    
    # Train model with class weights
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(train_data):
        X_train, X_val = train_data.iloc[train_idx], train_data.iloc[val_idx]
        y_train, y_val = train_labels.iloc[train_idx], train_labels.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, y_pred_proba))
    
    print(f"CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Train final model on all training data
    model.fit(train_data, train_labels)
    
    # Evaluation
    test_data = X[~train]
    test_labels = y[~train]
    
    if len(test_data) > 0:
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)[:, 1]
        
        auc = roc_auc_score(test_labels, probabilities)
        print(f'Test AUC = {auc:.4f}')
        
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
    
    # Save model
    model_name = f'{args.ticker}_market_simple_model.pkl'
    joblib.dump(model, model_name)
    print(f"\nModel saved as {model_name}")

if __name__ == '__main__':
    main() 
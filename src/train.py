import argparse, joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from .ingest import get_price
from .features import build_features
import pandas as pd, numpy as np

def label_next_day_return(df, threshold=0.01):
    """Метод для метки следующего дня, если доходность больше заданного порога."""
    # Since we're using auto_adjust=True in yfinance, we can use 'Close'
    # The price data has MultiIndex columns, so we need to access it properly
    if isinstance(df.columns, pd.MultiIndex):
        # For MultiIndex columns, we need to access the 'Close' column properly
        price_col = ('Close', df.columns.get_level_values(1)[0])  # Get the first ticker
    else:
        price_col = 'Close'
    
    # Calculate returns for different time horizons
    returns_1d = df[price_col].pct_change().shift(-1)
    returns_3d = df[price_col].pct_change(3).shift(-3)
    returns_5d = df[price_col].pct_change(5).shift(-5)
    
    # Create labels based on multiple criteria
    # 1. Short-term momentum (1-day return > threshold)
    # 2. Medium-term trend (3-day return > threshold/2)
    # 3. Volume confirmation (if available)
    
    label_1d = (returns_1d > threshold).astype(int)
    label_3d = (returns_3d > threshold/2).astype(int)
    
    # Combine labels - we want both short and medium term to be positive
    combined_label = ((label_1d == 1) & (label_3d == 1)).astype(int)
    
    return combined_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    # Загрузка только цен с помощью yfinance
    price = get_price(args.ticker, args.start, args.end)
    
    # Check if data was successfully downloaded
    if price.empty:
        print(f"Error: No data found for ticker '{args.ticker}'.")
        print("This could be because:")
        print("1. The ticker symbol is incorrect")
        print("2. The stock has been delisted")
        print("3. There's no data for the specified date range")
        print("4. Network connectivity issues")
        return
    
    # Check if we have enough data
    if len(price) < 100:
        print(f"Error: Insufficient data for ticker '{args.ticker}'.")
        print(f"Only {len(price)} data points found. Need at least 100 for training.")
        return
    
    # Новости нам не нужны, так что передаем пустой список
    feat = build_features(price, [])
    
    # Метка следующей доходности
    y = label_next_day_return(price)
    # Remove any non-feature columns if they exist, but don't assume specific columns
    columns_to_drop = []
    if 'date' in feat.columns:
        columns_to_drop.append('date')
    if 'Adj Close' in feat.columns:
        columns_to_drop.append('Adj Close')
    
    if columns_to_drop:
        X = feat.drop(columns=columns_to_drop)
    else:
        X = feat
    
    X, y = X.iloc[:-1], y.iloc[:-1]  # выравнивание длинны данных
    
    # Check if we have valid data after processing
    if X.empty or y.empty:
        print(f"Error: No valid data after processing for ticker '{args.ticker}'.")
        return
    
    if len(X) < 50:
        print(f"Error: Insufficient processed data for ticker '{args.ticker}'.")
        print(f"Only {len(X)} data points after processing. Need at least 50 for training.")
        return
    
    # Разделение на обучающий и тестовый наборы
    train = X.index < '2024-01-01'  # просто делаем разбиение
    
    # Check if we have enough training data
    train_data = X[train]
    train_labels = y[train]
    
    if len(train_data) < 20:
        print(f"Error: Insufficient training data for ticker '{args.ticker}'.")
        print(f"Only {len(train_data)} training samples found. Need at least 20 for training.")
        return
    
    if len(train_labels[train_labels == 1]) < 5 or len(train_labels[train_labels == 0]) < 5:
        print(f"Error: Insufficient class balance for ticker '{args.ticker}'.")
        print(f"Positive samples: {len(train_labels[train_labels == 1])}, Negative samples: {len(train_labels[train_labels == 0])}")
        print("Need at least 5 samples of each class for training.")
        return
    
    lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05)
    lgbm.fit(train_data, train_labels)
    
    # Оценка модели
    print('AUC test =', roc_auc_score(y[~train], lgbm.predict_proba(X[~train])[:,1]))
    
    # Сохранение модели
    joblib.dump(lgbm, f'{args.ticker}_model.pkl')

if __name__ == '__main__':
    main()

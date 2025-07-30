import argparse, joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np
import warnings
warnings.filterwarnings('ignore')

from .ingest import get_price
from .features import build_features
from .fundamentals import add_fundamental_features

def create_simple_labels(df, threshold=0.01):
    """Create simple yet effective labels."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    # Simple momentum with adaptive threshold
    returns = df[price_col].pct_change().shift(-1)
    
    # Use quantile-based threshold for better balance
    threshold = returns.quantile(0.6)  # Top 40% of returns
    
    labels = (returns > threshold).astype(int)
    
    # Ensure we return a Series
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    
    return labels

def add_simple_features(df):
    """Add simple yet effective features."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    # Price momentum
    df['price_momentum_1d'] = df[price_col].pct_change(1)
    df['price_momentum_3d'] = df[price_col].pct_change(3)
    df['price_momentum_5d'] = df[price_col].pct_change(5)
    
    # Volume features
    if 'Volume' in df.columns:
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_momentum'] = df['Volume'].pct_change()
    
    # Volatility
    returns = df[price_col].pct_change()
    df['volatility_20d'] = returns.rolling(20).std()
    
    # Price position
    df['price_position'] = (df[price_col] - df[price_col].rolling(20).min()) / \
                          (df[price_col].rolling(20).max() - df[price_col].rolling(20).min())
    
    # Time features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    # Load data
    price = get_price(args.ticker, args.start, args.end)
    
    if price.empty:
        print(f"Error: No data found for ticker '{args.ticker}'.")
        return
    
    if len(price) < 100:
        print(f"Error: Insufficient data for ticker '{args.ticker}'.")
        return
    
    # Build features
    feat = build_features(price, [])
    
    # Add simple features
    feat = add_simple_features(feat)
    
    # Add fundamental features
    feat = add_fundamental_features(feat, args.ticker)
    
    # Create labels
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
    
    if positive_ratio < 0.1 or positive_ratio > 0.9:
        print(f"Warning: Very imbalanced classes ({positive_ratio:.3f}). Consider adjusting threshold.")
    
    # Train/test split
    train = X.index < '2024-01-01'
    train_data = X[train]
    train_labels = y[train]
    
    if len(train_data) < 50:
        print(f"Error: Insufficient training data for ticker '{args.ticker}'.")
        return
    
    # Train model with class weights
    model = lgb.LGBMClassifier(
        n_estimators=300,
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
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    # Save model
    model_name = f'{args.ticker}_improved_model.pkl'
    joblib.dump(model, model_name)
    print(f"\nModel saved as {model_name}")

if __name__ == '__main__':
    main() 
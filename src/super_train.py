import argparse
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

from .ingest import get_price
from .market_enhanced_model import build_market_enhanced_features
from .fundamentals import add_fundamental_features
from .improved_labels import create_balanced_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    # Load price data
    price = get_price(args.ticker, args.start, args.end)
    if price.empty or len(price) < 100:
        print(f"Error: insufficient data for {args.ticker}")
        return

    # Build features with market context
    feat = build_market_enhanced_features(price, args.start, args.end)
    feat = add_fundamental_features(feat, args.ticker)

    # Create labels
    y = create_balanced_labels(price)

    # Drop potential non-feature columns
    drop_cols = [c for c in ['date', 'Adj Close'] if c in feat.columns]
    X = feat.drop(columns=drop_cols) if drop_cols else feat
    X, y = X.iloc[:-1], y.iloc[:-1]

    if X.empty or y.empty:
        print('Error: no valid data after processing')
        return

    # Train/test split
    train_mask = X.index < '2024-01-01'
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    if len(X_train) < 50:
        print('Error: insufficient training data')
        return

    # Base models
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, random_state=42, verbose=-1)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    lr = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('rf', rf), ('lr', lr)],
        voting='soft'
    )

    # Time series CV for metrics
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        ensemble.fit(X_tr, y_tr)
        probs = ensemble.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, probs))
    print(f'CV AUC: {pd.Series(cv_scores).mean():.4f}')

    # Fit on full training data
    ensemble.fit(X_train, y_train)

    if len(X_test) > 0:
        probs = ensemble.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        print(f'Test AUC = {auc:.4f}')
        print('\nClassification Report:')
        print(classification_report(y_test, preds))

    model_name = f'{args.ticker}_super_model.pkl'
    joblib.dump(ensemble, model_name)
    print(f'Model saved as {model_name}')


if __name__ == '__main__':
    main()

import argparse, joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .ingest import get_price
from .features import build_features
import pandas as pd, numpy as np
import warnings
warnings.filterwarnings('ignore')

def label_next_day_return(df, threshold=0.01):
    """Improved next-day labeling method."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    # Calculate returns for different time horizons
    returns_1d = df[price_col].pct_change().shift(-1)
    returns_3d = df[price_col].pct_change(3).shift(-3)
    returns_5d = df[price_col].pct_change(5).shift(-5)
    
    # Create labels based on multiple criteria
    label_1d = (returns_1d > threshold).astype(int)
    label_3d = (returns_3d > threshold/2).astype(int)
    
    # Combine labels - we want both short and medium term to be positive
    combined_label = ((label_1d == 1) & (label_3d == 1)).astype(int)
    
    return combined_label

def optimize_hyperparameters(X, y):
    """Hyperparameter optimization for LightGBM."""
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # LightGBM parameters to optimize
    param_grid = {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 63, 127],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Base model
    base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    
    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=tscv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def create_ensemble(X, y):
    """Create an ensemble of models."""
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Base models
    lgbm = lgb.LGBMClassifier(
        n_estimators=400, 
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    svm = SVC(probability=True, random_state=42)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('lgbm', lgbm),
            ('lr', lr),
            ('svm', svm)
        ],
        voting='soft'
    )
    
    return ensemble

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--optimize', action='store_true', help='Use hyperparameter optimization')
    ap.add_argument('--ensemble', action='store_true', help='Use ensemble model')
    args = ap.parse_args()

    # Load data
    price = get_price(args.ticker, args.start, args.end)
    
    # Check if data was successfully downloaded
    if price.empty:
        print(f"Error: No data found for ticker '{args.ticker}'.")
        return
    
    if len(price) < 100:
        print(f"Error: Insufficient data for ticker '{args.ticker}'.")
        return
    
    # Build features
    feat = build_features(price, [])
    y = label_next_day_return(price)
    
    # Remove any non-feature columns
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
    
    if len(X) < 50:
        print(f"Error: Insufficient processed data for ticker '{args.ticker}'.")
        return
    
    # Train/test split
    train = X.index < '2024-01-01'
    train_data = X[train]
    train_labels = y[train]
    
    if len(train_data) < 20:
        print(f"Error: Insufficient training data for ticker '{args.ticker}'.")
        return
    
    if len(train_labels[train_labels == 1]) < 5 or len(train_labels[train_labels == 0]) < 5:
        print(f"Error: Insufficient class balance for ticker '{args.ticker}'.")
        return
    
    # Model selection
    if args.optimize:
        print("Optimizing hyperparameters...")
        model = optimize_hyperparameters(train_data, train_labels)
    elif args.ensemble:
        print("Training ensemble model...")
        model = create_ensemble(train_data, train_labels)
        model.fit(train_data, train_labels)
    else:
        print("Training standard LightGBM model...")
        model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(train_data, train_labels)
    
    # Evaluation
    test_data = X[~train]
    test_labels = y[~train]
    
    if len(test_data) > 0:
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)[:, 1]
        
        auc = roc_auc_score(test_labels, probabilities)
        print(f'AUC test = {auc:.4f}')
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions))
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
    
    # Save model
    model_name = f'{args.ticker}_model.pkl'
    if args.optimize:
        model_name = f'{args.ticker}_optimized_model.pkl'
    elif args.ensemble:
        model_name = f'{args.ticker}_ensemble_model.pkl'
    
    joblib.dump(model, model_name)
    print(f"Model saved as {model_name}")

if __name__ == '__main__':
    main() 

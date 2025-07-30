import argparse, joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd, numpy as np
import warnings
warnings.filterwarnings('ignore')

from .ingest import get_price
from .features import build_features
from .market_data import get_market_data, calculate_market_features, get_sector_data, calculate_sector_features
from .simple_improved import add_simple_features, create_simple_labels

class MarketEnhancedPredictor:
    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.feature_importance = {}
    
    def create_models(self):
        """Create different models for the ensemble."""
        # LightGBM models with different focuses
        self.models['lgbm_momentum'] = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            class_weight='balanced', random_state=42, verbose=-1
        )
        
        self.models['lgbm_trend'] = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=8,
            class_weight='balanced', random_state=42, verbose=-1
        )
        
        # Random Forest for robustness
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=42
        )
        
        # Logistic Regression for linear relationships
        self.models['lr'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ])
    
    def train_models(self, X, y):
        """Train all models."""
        self.create_models()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X, y)
            
            # Cross-validation score
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                cv_scores.append(roc_auc_score(y_val, y_pred_proba))
            
            print(f"{name} CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
    
    def create_ensemble(self, X, y):
        """Create an ensemble of the trained models."""
        self.train_models(X, y)
        
        # Create ensemble with different weights
        estimators = []
        weights = []
        
        for name, model in self.models.items():
            estimators.append((name, model))
            # Give more weight to better performing models
            if 'lgbm' in name:
                weights.append(1.5)
            elif name == 'rf':
                weights.append(1.2)
            else:
                weights.append(1.0)
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
        # Train ensemble
        self.ensemble.fit(X, y)
        
        return self.ensemble
    
    def predict_with_confidence(self, X):
        """Predict with a confidence score."""
        predictions = {}
        
        # Individual model predictions
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]
            predictions[name] = pred_proba
        
        # Ensemble prediction
        ensemble_proba = self.ensemble.predict_proba(X)[:, 1]
        predictions['ensemble'] = ensemble_proba
        
        # Calculate confidence based on agreement
        individual_preds = np.array([pred for name, pred in predictions.items() if name != 'ensemble'])
        confidence = 1 - np.std(individual_preds, axis=0)
        
        return ensemble_proba, confidence, predictions

def build_market_enhanced_features(price_df, start_date, end_date):
    """Build features that include market context."""
    print("Building enhanced features with market data...")
    
    # Get market data
    market_data = get_market_data(start_date, end_date)
    
    # Get sector data
    ticker = price_df.columns.get_level_values(1)[0] if isinstance(price_df.columns, pd.MultiIndex) else 'AAPL'
    sector_data, sector_name = get_sector_data(ticker, start_date, end_date)
    
    # Build base features
    feat = build_features(price_df, [])
    
    # Add simple features
    feat = add_simple_features(feat)
    
    # Add market features
    market_features = calculate_market_features(market_data, price_df)
    feat = pd.concat([feat, market_features], axis=1)
    
    # Add sector features
    if sector_data is not None:
        sector_features = calculate_sector_features(sector_data, price_df)
        feat = pd.concat([feat, sector_features], axis=1)
        print(f"✓ Added sector features for {sector_name}")
    
    # Fill missing values
    feat = feat.ffill().fillna(0)
    
    print(f"✓ Total features: {len(feat.columns)}")
    return feat

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
    
    # Build enhanced features with market data
    feat = build_market_enhanced_features(price, args.start, args.end)
    
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
    
    # Train/test split
    train = X.index < '2024-01-01'
    train_data = X[train]
    train_labels = y[train]
    
    if len(train_data) < 50:
        print(f"Error: Insufficient training data for ticker '{args.ticker}'.")
        return
    
    # Create and train ensemble
    predictor = MarketEnhancedPredictor()
    ensemble = predictor.create_ensemble(train_data, train_labels)
    
    # Evaluation
    test_data = X[~train]
    test_labels = y[~train]
    
    if len(test_data) > 0:
        # Ensemble predictions
        ensemble_proba, confidence, all_predictions = predictor.predict_with_confidence(test_data)
        
        # Metrics
        auc = roc_auc_score(test_labels, ensemble_proba)
        print(f'\nEnsemble AUC: {auc:.4f}')
        
        # Classification report
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        print("\nClassification Report:")
        print(classification_report(test_labels, ensemble_pred))
        
        # Feature importance
        if predictor.feature_importance:
            print("\nTop Features by Model:")
            for model_name, importance_df in predictor.feature_importance.items():
                print(f"\n{model_name}:")
                print(importance_df.head(5))
    
    # Save model
    model_name = f'{args.ticker}_market_enhanced_model.pkl'
    joblib.dump(predictor, model_name)
    print(f"\nModel saved as {model_name}")

if __name__ == '__main__':
    main() 
import argparse, joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd, numpy as np
import warnings
warnings.filterwarnings('ignore')

from .ingest import get_price
from .enhanced_features import build_enhanced_features
from .improved_labels import create_balanced_labels, create_adaptive_threshold

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def create_models(self):
        """Создание различных моделей для ансамбля."""
        # LightGBM with different parameters
        self.models['lgbm_fast'] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=42, verbose=-1
        )
        
        self.models['lgbm_deep'] = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=8,
            random_state=42, verbose=-1
        )
        
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )
        
        # Logistic Regression with scaling
        self.models['lr'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # SVM with scaling
        self.models['svm'] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True, random_state=42))
        ])
    
    def train_models(self, X, y):
        """Обучение всех моделей."""
        self.create_models()
        
        # Time series split for validation
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
        """Создание ансамбля моделей."""
        self.train_models(X, y)
        
        # Create ensemble
        estimators = []
        for name, model in self.models.items():
            estimators.append((name, model))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[1, 1, 1, 1, 1]  # Equal weights
        )
        
        # Train ensemble
        self.ensemble.fit(X, y)
        
        return self.ensemble
    
    def predict_with_confidence(self, X):
        """Предсказание с уровнем уверенности."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--label_strategy', choices=['balanced', 'adaptive', 'momentum'], default='balanced')
    args = ap.parse_args()

    # Загрузка данных
    price = get_price(args.ticker, args.start, args.end)
    
    if price.empty:
        print(f"Error: No data found for ticker '{args.ticker}'.")
        return
    
    # Построение расширенных признаков
    feat = build_enhanced_features(price, [])
    
    # Создание меток в зависимости от стратегии
    if args.label_strategy == 'balanced':
        y = create_balanced_labels(price)
    elif args.label_strategy == 'adaptive':
        y = create_adaptive_threshold(price)
    else:
        # Simple momentum
        if isinstance(price.columns, pd.MultiIndex):
            price_col = ('Close', price.columns.get_level_values(1)[0])
        else:
            price_col = 'Close'
        returns = price[price_col].pct_change().shift(-1)
        y = (returns > 0.01).astype(int)
    
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
    
    # Train/test split
    train = X.index < '2024-01-01'
    train_data = X[train]
    train_labels = y[train]
    
    if len(train_data) < 50:
        print(f"Error: Insufficient training data for ticker '{args.ticker}'.")
        return
    
    # Create and train ensemble
    predictor = StockPredictor()
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
    model_name = f'{args.ticker}_ensemble_{args.label_strategy}.pkl'
    joblib.dump(predictor, model_name)
    print(f"\nModel saved as {model_name}")

if __name__ == '__main__':
    main() 
import pandas as pd
import numpy as np

def create_balanced_labels(df, threshold=0.01):
    """Create balanced labels using multiple strategies."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    # Strategy 1: Simple momentum
    returns_1d = df[price_col].pct_change().shift(-1)
    momentum_label = (returns_1d > threshold).astype(int)
    
    # Strategy 2: Breakout detection
    sma_20 = df[price_col].rolling(20).mean()
    breakout_label = (df[price_col] > sma_20 * 1.02).astype(int)
    
    # Strategy 3: Volume confirmation
    if 'Volume' in df.columns:
        volume_ma = df['Volume'].rolling(20).mean()
        volume_spike = (df['Volume'] > volume_ma * 1.5).astype(int)
    else:
        volume_spike = pd.Series(0, index=df.index)
    
    # Strategy 4: RSI oversold/overbought
    if 'RSI_14' in df.columns:
        rsi_signal = ((df['RSI_14'] < 30) | (df['RSI_14'] > 70)).astype(int)
    else:
        rsi_signal = pd.Series(0, index=df.index)
    
    # Combine strategies with weights
    combined_score = (
        momentum_label * 0.4 +
        breakout_label * 0.3 +
        volume_spike * 0.2 +
        rsi_signal * 0.1
    )
    
    # Create balanced labels
    final_label = (combined_score > 0.5).astype(int)
    
    # Ensure we return a Series, not DataFrame
    if isinstance(final_label, pd.DataFrame):
        final_label = final_label.iloc[:, 0]
    
    return final_label

def create_adaptive_threshold(df, target_positive_ratio=0.3):
    """Create an adaptive threshold for balanced labels."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    returns = df[price_col].pct_change().shift(-1)
    
    # Find threshold that gives target positive ratio
    sorted_returns = returns.dropna().sort_values()
    target_index = int(len(sorted_returns) * (1 - target_positive_ratio))
    adaptive_threshold = sorted_returns.iloc[target_index]
    
    labels = (returns > adaptive_threshold).astype(int)
    
    # Ensure we return a Series, not DataFrame
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    
    return labels

def create_multi_horizon_labels(df):
    """Create labels for multiple time horizons."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    # Different time horizons
    horizons = [1, 3, 5, 10]
    labels = {}
    
    for horizon in horizons:
        returns = df[price_col].pct_change(horizon).shift(-horizon)
        # Use adaptive threshold for each horizon
        threshold = returns.quantile(0.7)  # Top 30%
        labels[f'label_{horizon}d'] = (returns > threshold).astype(int)
    
    return labels 

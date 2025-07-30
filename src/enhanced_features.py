import pandas as pd
import numpy as np
from .technicals import add_technicals

def add_market_regime_features(df):
    """Add market regime features."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    # Volatility regime
    returns = df[price_col].pct_change()
    volatility = returns.rolling(20).std()
    high_vol = (volatility > volatility.quantile(0.8)).astype(int)
    low_vol = (volatility < volatility.quantile(0.2)).astype(int)
    
    # Trend regime
    sma_20 = df[price_col].rolling(20).mean()
    sma_50 = df[price_col].rolling(50).mean()
    uptrend = (sma_20 > sma_50).astype(int)
    downtrend = (sma_20 < sma_50).astype(int)
    
    # Price position relative to range
    high_20 = df[price_col].rolling(20).max()
    low_20 = df[price_col].rolling(20).min()
    price_position = (df[price_col] - low_20) / (high_20 - low_20)
    
    # Add features - ensure they are Series
    df['high_volatility'] = high_vol.astype(int)
    df['low_volatility'] = low_vol.astype(int)
    df['uptrend'] = uptrend.astype(int)
    df['downtrend'] = downtrend.astype(int)
    df['price_position'] = price_position.fillna(0)
    
    return df

def add_time_features(df):
    """Add time-based features."""
    # Day of week
    df['day_of_week'] = df.index.dayofweek
    
    # Month
    df['month'] = df.index.month
    
    # Quarter
    df['quarter'] = df.index.quarter
    
    # Is month end
    df['is_month_end'] = df.index.is_month_end.astype(int)
    
    # Is quarter end
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    return df

def add_lag_features(df, lags=[1, 2, 3, 5, 10]):
    """Add lag features."""
    if isinstance(df.columns, pd.MultiIndex):
        price_col = ('Close', df.columns.get_level_values(1)[0])
    else:
        price_col = 'Close'
    
    # Price lags
    for lag in lags:
        df[f'price_lag_{lag}'] = df[price_col].shift(lag)
        df[f'return_lag_{lag}'] = df[price_col].pct_change(lag)
    
    # Volume lags (if available)
    if 'Volume' in df.columns:
        for lag in lags:
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
    
    return df

def add_interaction_features(df):
    """Add interaction features."""
    # Price * Volume interaction
    if 'Volume' in df.columns:
        df['price_volume'] = df['Close'] * df['Volume']
    
    # RSI * Volume interaction
    if 'RSI_14' in df.columns and 'Volume' in df.columns:
        df['rsi_volume'] = df['RSI_14'] * df['Volume']
    
    # MACD * Price interaction
    if 'MACD_12_26_9' in df.columns:
        df['macd_price'] = df['MACD_12_26_9'] * df['Close']
    
    return df

def build_enhanced_features(price_df, news_list=[]):
    """Build extended feature set."""
    # Start with technical indicators
    feat = add_technicals(price_df)
    
    # Add market regime features
    feat = add_market_regime_features(feat)
    
    # Add time features
    feat = add_time_features(feat)
    
    # Add lag features
    feat = add_lag_features(feat)
    
    # Add interaction features
    feat = add_interaction_features(feat)
    
    # Handle news sentiment if available
    if news_list:
        # ... existing news processing code ...
        pass
    
    # Fill missing values
    feat = feat.ffill().fillna(0)
    
    return feat 
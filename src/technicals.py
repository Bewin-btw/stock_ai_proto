import pandas as pd, pandas_ta as ta

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators."""
    df = df.copy()

    # Switch from MultiIndex to a regular DataFrame
    # Get the first ticker from the MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        ticker = df.columns.get_level_values(1)[0]  # Get the first ticker
        df = df.xs(ticker, level='Ticker', axis=1)

    # Add technical indicators
    # Moving averages
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True)
    
    # MACD
    df.ta.macd(append=True)
    
    # RSI
    df.ta.rsi(append=True)
    df.ta.rsi(length=14, append=True)
    
    # Bollinger Bands
    df.ta.bbands(append=True)
    
    # Stochastic
    df.ta.stoch(append=True)
    
    # ATR
    df.ta.atr(length=14, append=True)
    
    # Volume indicators
    df.ta.obv(append=True)
    df.ta.ad(append=True)
    
    # Price action
    df.ta.hlc3(append=True)
    # df.ta.typical_price(append=True)  # Not available in this version
    
    # Momentum
    df.ta.mom(append=True)
    df.ta.roc(append=True)
    
    return df

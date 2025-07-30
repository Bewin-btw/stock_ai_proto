import pandas as pd
import yfinance as yf
import numpy as np

def get_market_data(start_date, end_date):
    """Fetch market data for broader context."""
    # Major indices
    indices = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ ETF', 
        'IWM': 'Russell 2000 ETF',
        'VIX': 'Volatility Index',
        'GLD': 'Gold ETF',
        'TLT': 'Bond ETF'
    }
    
    market_data = {}
    
    for ticker, description in indices.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if not data.empty:
                market_data[ticker] = data
                print(f"✓ Downloaded {ticker} ({description})")
            else:
                print(f"✗ Failed to download {ticker}")
        except Exception as e:
            print(f"✗ Error downloading {ticker}: {e}")
    
    return market_data

def calculate_market_features(market_data, stock_data):
    """Calculate market features."""
    features = pd.DataFrame(index=stock_data.index)
    
    # SPY (S&P 500) features
    if 'SPY' in market_data:
        spy = market_data['SPY']['Close']
        features['spy_momentum'] = spy.pct_change(5)
        features['spy_volatility'] = spy.pct_change().rolling(20).std()
        features['spy_trend'] = (spy > spy.rolling(50).mean()).astype(int)
    
    # QQQ (NASDAQ) features
    if 'QQQ' in market_data:
        qqq = market_data['QQQ']['Close']
        features['qqq_momentum'] = qqq.pct_change(5)
        features['tech_trend'] = (qqq > qqq.rolling(50).mean()).astype(int)
    
    # VIX (Volatility) features
    if 'VIX' in market_data:
        vix = market_data['VIX']['Close']
        features['market_fear'] = (vix > vix.rolling(20).mean()).astype(int)
        features['vix_level'] = vix / vix.rolling(252).mean()  # Normalized VIX
    
    # Gold features
    if 'GLD' in market_data:
        gld = market_data['GLD']['Close']
        features['gold_momentum'] = gld.pct_change(5)
        features['flight_to_safety'] = (gld > gld.rolling(20).mean()).astype(int)
    
    # Bond features
    if 'TLT' in market_data:
        tlt = market_data['TLT']['Close']
        features['bond_momentum'] = tlt.pct_change(5)
        features['risk_off'] = (tlt > tlt.rolling(20).mean()).astype(int)
    
    # Market correlations
    if 'SPY' in market_data and len(stock_data) > 20:
        stock_returns = stock_data['Close'].pct_change()
        spy_returns = market_data['SPY']['Close'].pct_change()
        
        # Rolling correlation
        correlation = stock_returns.rolling(20).corr(spy_returns)
        if isinstance(correlation, pd.DataFrame):
            correlation = correlation.iloc[:, 0]  # Take first column if DataFrame
        features['market_correlation'] = correlation
    
    # Fill missing values
    features = features.fillna(0)
    
    return features

def get_sector_data(ticker, start_date, end_date):
    """Fetch sector data."""
    # Sector ETFs mapping
    sector_etfs = {
        'XLK': 'Technology',
        'XLF': 'Financials', 
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLB': 'Materials'
    }
    
    # Determine which sector to use based on ticker
    # This is a simplified mapping - in practice you'd use a proper sector classification
    sector_mapping = {
        'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLK', 'NVDA': 'XLK', 'TSLA': 'XLK',
        'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF',
        'XOM': 'XLE', 'CVX': 'XLE',
        'JNJ': 'XLV', 'PFE': 'XLV',
        'GE': 'XLI', 'CAT': 'XLI',
        'PG': 'XLP', 'KO': 'XLP',
        'AMZN': 'XLY', 'HD': 'XLY',
        'DUK': 'XLU', 'SO': 'XLU',
        'SPG': 'XLRE', 'PLD': 'XLRE',
        'FCX': 'XLB', 'NEM': 'XLB'
    }
    
    sector_etf = sector_mapping.get(ticker, 'XLK')  # Default to tech
    
    try:
        sector_data = yf.download(sector_etf, start=start_date, end=end_date, auto_adjust=True)
        if not sector_data.empty:
            return sector_data, sector_etfs[sector_etf]
        else:
            return None, None
    except:
        return None, None

def calculate_sector_features(sector_data, stock_data):
    """Calculate sector-related features."""
    features = pd.DataFrame(index=stock_data.index)
    
    if sector_data is not None:
        sector_close = sector_data['Close']
        
        # Sector momentum
        features['sector_momentum'] = sector_close.pct_change(5)
        features['sector_trend'] = (sector_close > sector_close.rolling(50).mean()).astype(int)
        
        # Stock vs Sector performance
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_close = stock_data[('Close', stock_data.columns.get_level_values(1)[0])]
        else:
            stock_close = stock_data['Close']
        features['vs_sector_performance'] = stock_close.pct_change(5) - sector_close.pct_change(5)
        
        # Sector volatility
        features['sector_volatility'] = sector_close.pct_change().rolling(20).std()
    
    features = features.fillna(0)
    return features 
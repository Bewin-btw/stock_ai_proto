import yfinance as yf
import pandas as pd

def get_fundamentals(ticker):
    """Fetch fundamental metrics via `yfinance.Ticker.info`."""
    info = yf.Ticker(ticker).info
    # Collect only relevant metrics (when available)
    keys = [
        'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months',
        'returnOnEquity', 'returnOnAssets', 'debtToEquity', 'dividendYield',
        'marketCap', 'enterpriseValue', 'profitMargins', 'grossMargins',
        'operatingMargins', 'revenueGrowth', 'earningsGrowth', 'ebitdaMargins',
        'trailingEps', 'forwardEps', 'pegRatio', 'beta', 'bookValue', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow'
    ]
    fundamentals = {k: info.get(k, None) for k in keys}
    return fundamentals

def add_fundamental_features(df, ticker):
    """Add fundamental features to the DataFrame (same for all dates)."""
    fundamentals = get_fundamentals(ticker)
    for k, v in fundamentals.items():
        df[f'fund_{k}'] = v if v is not None else 0
    return df

if __name__ == '__main__':
    # Example usage
    print(get_fundamentals('AAPL'))
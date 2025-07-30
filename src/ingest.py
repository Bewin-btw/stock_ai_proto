import os
import pandas as pd
import yfinance as yf


def get_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch price data via yfinance."""
    return yf.download(ticker, start=start, end=end, auto_adjust=True, threads=False)


def get_news(ticker: str, from_dt: str) -> list:
    """Return news items from yfinance if available."""
    try:
        news = yf.Ticker(ticker).news
        # Filter by date
        from_ts = pd.Timestamp(from_dt).timestamp()
        filtered = [n for n in news if n.get('providerPublishTime', 0) >= from_ts]
        return filtered
    except Exception as e:
        print(f"Warning: could not fetch news for {ticker}: {e}")
        return []

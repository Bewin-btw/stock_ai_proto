import os
import pandas as pd
import yfinance as yf

def get_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Получение данных о ценах акций через yfinance."""
    return yf.download(ticker, start=start, end=end, auto_adjust=True, threads=False)

# Получение новостей через yfinance.Ticker.news
# Возвращает список словарей с ключами: 'title', 'publisher', 'link', 'providerPublishTime', 'type', 'content'
def get_news(ticker: str, from_dt: str) -> list:
    """Получение новостей через yfinance (если доступно)."""
    try:
        news = yf.Ticker(ticker).news
        # Фильтруем по дате
        from_ts = pd.Timestamp(from_dt).timestamp()
        filtered = [n for n in news if n.get('providerPublishTime', 0) >= from_ts]
        return filtered
    except Exception as e:
        print(f"Warning: could not fetch news for {ticker}: {e}")
        return []

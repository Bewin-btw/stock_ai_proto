import os
import pandas as pd
import yfinance as yf

def get_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Получение данных о ценах акций через yfinance."""
    return yf.download(ticker, start=start, end=end, auto_adjust=True, threads=False)

# Если новостей нет, можно оставить метод заглушки
def get_news(query: str, from_dt: str) -> list:
    """Заглушка для новостей, если не используем внешний API."""
    return []

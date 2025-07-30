import pandas as pd
from .technicals import add_technicals
from .sentiment import score_texts
import json
import numpy as np

def aggregate_news_sentiment(news_list, index):
    """Агрегирует сентимент по датам, возвращает DataFrame с датами и признаками сентимента."""
    if not news_list:
        return pd.DataFrame(index=index)
    news_df = pd.DataFrame(news_list)
    # Преобразуем timestamp в дату
    news_df['date'] = pd.to_datetime(news_df['providerPublishTime'], unit='s').dt.date
    # Считаем сентимент
    news_df['sentiment'] = score_texts(news_df['title'].fillna(''))
    # Группируем по дате
    daily = news_df.groupby('date').agg(
        avg_sentiment=('sentiment', 'mean'),
        pos_news=('sentiment', lambda x: (x > 0).sum()),
        neg_news=('sentiment', lambda x: (x < 0).sum()),
        news_count=('sentiment', 'count')
    )
    # Приводим индекс к DatetimeIndex
    daily.index = pd.to_datetime(daily.index)
    # Ресемплируем на индекс цен
    daily = daily.reindex(index, method='ffill').fillna(0)
    # rolling-агрегация за 3 и 7 дней
    daily['avg_sentiment_3d'] = daily['avg_sentiment'].rolling(3, min_periods=1).mean()
    daily['avg_sentiment_7d'] = daily['avg_sentiment'].rolling(7, min_periods=1).mean()
    daily['pos_news_3d'] = daily['pos_news'].rolling(3, min_periods=1).sum()
    daily['neg_news_3d'] = daily['neg_news'].rolling(3, min_periods=1).sum()
    daily['news_count_3d'] = daily['news_count'].rolling(3, min_periods=1).sum()
    daily['pos_news_7d'] = daily['pos_news'].rolling(7, min_periods=1).sum()
    daily['neg_news_7d'] = daily['neg_news'].rolling(7, min_periods=1).sum()
    daily['news_count_7d'] = daily['news_count'].rolling(7, min_periods=1).sum()
    return daily

def build_features(price_df: pd.DataFrame, news_list: list[dict]) -> pd.DataFrame:
    # Technicals
    feat = add_technicals(price_df)
    
    # Сентимент новостей (агрегированный)
    if news_list:
        sentiment_df = aggregate_news_sentiment(news_list, feat.index)
        feat = pd.concat([feat, sentiment_df], axis=1)
    
    feat = feat.ffill().fillna(0)
    return feat

def label_next_day_return(df: pd.DataFrame) -> pd.Series:
    # Используем 'close' вместо 'Adj Close'
    returns = df['close'].pct_change().shift(-1)
    return returns

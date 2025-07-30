import pandas as pd
from .technicals import add_technicals
from .sentiment import score_texts
import json

def build_features(price_df: pd.DataFrame, news_list: list[dict]) -> pd.DataFrame:
    # Technicals
    feat = add_technicals(price_df)
    
    # Sentiment aggregated daily
    if news_list:
        news_df = pd.DataFrame(news_list)
        
        # Преобразуем строку в словарь для каждого элемента в колонке 'feed'
        news_df['feed'] = news_df['feed'].apply(json.loads)
        
        # Извлекаем дату публикации
        news_df['publishedAt'] = news_df['feed'].apply(lambda x: pd.to_datetime(x['time_published'], errors='coerce')).dt.date
        
        # Обрабатываем сентимент
        news_df['sent'] = score_texts(news_df['title'] + '. ' + news_df['description'].fillna(''))
        
        # Группируем по дате
        daily_sent = news_df.groupby('publishedAt')['sent'].mean()
        
        # Слияние с основным фреймом данных
        feat['date'] = feat.index.date
        feat = feat.merge(daily_sent, left_on='date', right_index=True, how='left')
    
    feat = feat.ffill().fillna(0)
    return feat


def label_next_day_return(df: pd.DataFrame) -> pd.Series:
    # Используем 'close' вместо 'Adj Close'
    returns = df['close'].pct_change().shift(-1)
    return returns

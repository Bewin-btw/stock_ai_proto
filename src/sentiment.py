from transformers import pipeline
import pandas as pd

_sent_clf = None
def _load():
    global _sent_clf
    if _sent_clf is None:
        _sent_clf = pipeline(
            task='sentiment-analysis',
            model='AI4Finance-Foundation/FinGPT-Sentiment-Base'
        )
    return _sent_clf

def score_texts(texts: list[str]) -> pd.Series:
    clf = _load()
    preds = clf(texts, truncation=True)
    mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    return pd.Series([mapping[p['label']] * p['score'] for p in preds])

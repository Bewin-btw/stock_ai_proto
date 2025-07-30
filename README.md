# Stock AI Prototype

Minimal end‑to‑end pipeline to generate buy/sell signals from
market data **+** news/social sentiment.

```
pip install -r requirements.txt
python -m src.train --ticker AAPL --start 2020-01-01 --end 2025-01-01
python -m src.predict --ticker AAPL
```

## Modules
| file | purpose |
|------|---------|
| `ingest.py` | download price + news |
| `technicals.py` | compute TA indicators (`pandas_ta`) |
| `sentiment.py` | FinGPT/FinBERT sentiment scoring |
| `features.py` | assemble feature matrix for ML |
| `train.py` | LightGBM classifier + backtest |
| `predict.py` | load model & produce latest signal |
| `risk.py` | simple Kelly sizing & VaR guard |


## Advanced usage
To train the enhanced model with market and fundamental data:
```
python -m src.super_train --ticker AAPL --start 2020-01-01 --end 2025-01-01
python -m src.super_predict --ticker AAPL
```

# Stock AI Prototype

Minimal end‑to‑end pipeline to generate buy/sell signals from
market data **+** news/social sentiment.

```
pip install -r requirements.txt
python -m src.train --ticker AAPL --start 2020-01-01 --end 2025-01-01
python -m src.predict --ticker AAPL
python -m src.deep_learning_model --ticker AAPL \
  --start 2020-01-01 --end 2024-12-31 \
  --epochs 20 --hidden 128 --layers 3
python -m src.deep_predict --ticker AAPL --start 2024-01-01 --end 2025-01-01
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
| `deep_learning_model.py` | LSTM model with dropout and early stopping |
| `deep_predict.py` | run saved LSTM model to get latest signal |
| `risk.py` | simple Kelly sizing & VaR guard |

Fill in API keys in `.env` or as env‑vars before running.
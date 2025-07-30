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
# the checkpoint stores all hyperparameters so prediction uses the same model

python -m src.transformer_model --ticker AAPL \
  --start 2020-01-01 --end 2024-12-31 \
  --epochs 20 --heads 4 --layers 2
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
| `transformer_model.py` | Transformer-based price sequence model |
| `risk.py` | simple Kelly sizing & VaR guard |

Fill in API keys in `.env` or as env‑vars before running.

## Parameter tips

- `ticker`: stock symbol, e.g. `AAPL`
- `start` / `end`: date range for data in `YYYY-MM-DD` format
- `epochs`: number of training iterations
- `hidden`: hidden state size for the LSTM
- `layers`: number of stacked layers in LSTM or Transformer
- `heads`: attention heads for the Transformer model

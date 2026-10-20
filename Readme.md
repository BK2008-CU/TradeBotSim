# Alpaca Signal Bot

A Streamlit dashboard that scans a broad stock universe and gives each ticker a
BUY / HOLD / SELL signal from three sources:

1. **Technical** (trend, RSI, MACD, Bollinger Bands)
2. **ML model** (gradient boosting predicting 5-day direction, shrunk toward 0 if it can't beat a coin flip on a holdout set)
3. **News sentiment** (VADER scoring of recent Alpaca news headlines)

You control the weights and thresholds in the sidebar. Orders go to your Alpaca
**paper** account only unless you change `ALPACA_PAPER`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your Alpaca paper keys into .env
streamlit run app.py
```

Get free paper-trading keys at https://alpaca.markets (Paper Trading > API Keys).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (Scanner, Deep dive, Portfolio tabs) |
| `signals.py` | Indicators, ML model, sentiment, signal combination |
| `market_data.py` | Alpaca price bars and news |
| `broker.py` | Alpaca account, positions, orders |
| `universe.py` | Sector-grouped ticker list (edit to taste) |

## Limitations

- Uses Alpaca's free IEX data feed; prices and volume can differ slightly from consolidated data.
- News covers recent headlines only. There is no earnings calendar or analyst-forecast data.
- Short-term price prediction is very hard. Treat signals as one input, not advice.
- Stay on paper trading until you've backtested the strategy.
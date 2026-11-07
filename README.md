# Kalshi Autoquoter Framework

Pricing and quoting code for binary contracts on FX and rates, written against the Kalshi API.

**Placeholder parameters.** Edge, requote threshold, sizing and budget values in this repo are placeholders chosen for illustration, not the values used in live trading. `DRY_RUN = True` by default, so nothing places orders until you change it.

## Contents

| File | Content |
|---|---|
| `EURUSDPricer.py`, `DXYPricer.py` | Fair value for "above/below strike at settlement" contracts: a GARCH volatility fit to recent intraday bars and Black-Scholes digital pricing, with a Pyth Hermes spot feed. |
| `eurusd_ladder_autoquoter.py` | Quotes a whole strike ladder from one shared spot fetch and one shared GARCH fit per loop, with a pooled budget across strikes. |
| `dxy_autoquoter.py` | Single-contract version of the same loop. |
| `autoquoter.py` | Instrument-agnostic quoting class that takes a `price_fn` returning fair yes/no cents. |
| `kalshi_treasury_lines.py`, `kalshi_treasury_fair_value.py`, `kalshi_treasury_quote.py` | Treasury-yield contracts: discover lines by tenor, price them from a calibrated CIR model (`cir_three_way_comparison.py`, `cir_european_chain.py`, `calibrate_treasury.py`, `live_treasury_yields_pyth.py`, `bond_price_yield_pyth.py`), screen by model fair value, and quote with a subclass of `Autoquoter`. |
| `KalshiClient.py` | Signed REST client (RSA request signing) for markets, orders and portfolio. |

## Quoting logic

- Edge shrinks linearly with distance from the 50-cent point, with a floor.
- Only the in-the-money side is quoted; the other side is pulled.
- Orders are post-only. A side is requoted only when its target moves past a threshold.
- Exposure is capped by a hard budget that counts filled and resting size.

## Setup

```bash
pip install numpy pandas scipy arch requests cryptography
export KALSHI_KEY_ID=...                 # your own key
export KALSHI_PRIVATE_KEY_PATH=...       # path to your own key file
```

No credentials or market data are included. For research and education only. Not investment advice.

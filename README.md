# Options GEX Analytics

BOVA11 — B3 Brazilian options via COTAHIST + OI proxy

## What it does

- Global and range-based Put/Call Ratio
- IV skew (OTM puts vs OTM calls)
- Notional by strike (volume financeiro)
- Gamma Exposure (Customer/Dealer)
- Call/Put walls and Gamma Flip
- $IND ↔ BOVA11 Kalman regression & delta-neutral hedge sizing

## Requirements

**Python 3.10** (64-bit) — required by MetaTrader 5. The MT5 Python integration only supports the **64-bit** build. Install it from [python.org](https://www.python.org/downloads/release/python-3100/) and make sure "Add Python to PATH" is checked.

### Dependencies

```
pip install numpy pandas scipy matplotlib requests MetaTrader5
```

| Package | Purpose |
|---|---|
| `numpy` | Array operations, Greeks computation |
| `pandas` | DataFrames for option chains and COTAHIST data |
| `scipy` | Black-Scholes pricing (stats, optimize) |
| `matplotlib` | GEX and notional charts |
| `requests` | Downloading B3 COTAHIST files |
| `MetaTrader5` | Live market data and order execution via MT5 terminal |

### MetaTrader 5 setup

1. Install MetaTrader 5 and log in to a broker account.
2. In MT5, go to **Tools → Options → Expert Advisors** and enable **Allow Algo Trading**.
3. Place this project folder under your MT5 `Scripts` directory (e.g. `MQL5/Scripts/GEX_Analytics_options_futures`).
4. Run from a terminal or from within MT5's integrated Python environment.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Main analysis & entry point (`analyze_options` + `main`) |
| `bs_greeks.py` | Black-Scholes pricing, Greeks, implied vol |
| `gex_utils.py` | Gamma flip detection |
| `gex_plots.py` | All matplotlib charts (notional, Friday GEX, all-expiry GEX) |
| `b3_options_loader.py` | B3 COTAHIST data fetch, call/put classification, Greek computation |
| `kalman_price_mapper.py` | Kalman filter $IND ↔ BOVA11 mapping & delta-neutral hedge sizing |
| `constants.py` | Shared constants (`ASSET_SYMBOL`, `PERIODS`, etc.) |
| `mt5_connector.py` | MetaTrader 5 data connector |

## Practical usage — Intraday $IND trading

### Best days to run

| Day | All-expiry GEX reliability | Friday-only GEX | Best use |
|---|---|---|---|
| **Monday** | Most reliable | Plan weekly hedges | Set the week's key levels |
| **Tuesday** | Very reliable | Confirm Monday's levels | Validate / adjust positions |
| **Wednesday** | Good | Weekly gamma building | Mid-week check |
| **Thursday** | Degrading | High gamma, unstable | Watch for pin risk |
| **Friday** | Noisy | Expiration gamma spike | Intraday only |

**Mon/Tue** give the most reliable GEX levels — the full gamma profile is intact after Friday's expiry clears out. By Thursday/Friday, short-dated gamma dominates and walls become unstable; lean on the Friday-specific GEX section instead.

### Recommended intraday timeframe: 15-minute bars

- Dealer hedging rebalances are visible at this granularity.
- Clean wall tests: call wall = resistance, put wall = support.
- Drop to **5-min** if spot is within ±0.5% of the gamma flip (transition zone).

### Session workflow

1. **Pre-market (09:00 BRT):** Run the script → note `$IND` call wall, put wall, gamma flip.
2. **10:00–11:30:** First 6 fifteen-minute bars — price discovery vs GEX levels.
3. **Wall touch on 15-min close** → mean-reversion entry (positive gamma regime: spot below flip).
4. **Wall break on 15-min close** → trend continuation (negative gamma regime: spot above flip).
5. **14:00–16:00:** Strongest dealer hedging flow period; 15-min signals at GEX levels are most reliable.

### Gamma regime quick reference

**Spot Price Relative to Gamma Flip for Positive Gamma:**

The spot price must be above the gamma flip level for the market to be in a positive gamma (positive GEX) regime.

- **When spot > gamma flip:** Market makers are net long gamma.
	- Their hedging behavior is counter-cyclical (buying dips, selling rallies), which dampens volatility and promotes range-bound, mean-reverting price action.
- **When spot < gamma flip:** Market makers are net short gamma.
	- Hedging becomes pro-cyclical (buying rallies, selling crashes), amplifying volatility and enabling trending moves.

**Key Insight:** The gamma flip (or zero gamma level) is the threshold where dealer hedging transitions from stabilizing to destabilizing. Traders monitor whether price is above or below this level to determine the prevailing volatility regime.

| Regime         | Condition                | Dealer behavior         | Strategy                                                      |
|---------------|--------------------------|-------------------------|---------------------------------------------------------------|
| Positive gamma| Spot above gamma flip    | Dealers dampen moves    | Mean-reversion at walls (buy put wall, sell call wall)         |
| Negative gamma| Spot below gamma flip    | Dealers amplify moves   | Sell below gamma flip (trend continuation on wall breaks)      |
| Transition zone| Spot within ±0.5% of flip| Unstable                | Reduce size, use 5-min confirmation                            |

**Key levels:**
- **Gamma flip:** Buy above gamma flip (bullish regime), sell below (trend regime).
- **Call wall:** Sell/short at call wall (resistance).
- **Put wall:** Buy/long at put wall (support).

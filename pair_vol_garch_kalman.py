# -*- coding: utf-8 -*-
"""
Pair-Trading Volatility Estimator — WINJ26 × WDOJ26
====================================================
Estimates the volatility of the spread between Mini Ibovespa (WINJ26) and
Mini Dollar (WDOJ26) futures on B3, using:

  1. GARCH(1,1) on the spread log-returns
  2. Kalman Filter for dynamic hedge-ratio β and spread residual volatility

Outputs
-------
- Current GARCH(1,1) conditional volatility (annualised & daily)
- Kalman Filter spread volatility (from innovation series)
- Plots: spread time series, rolling volatility, GARCH vs Kalman comparison

Usage
-----
    python pair_vol_garch_kalman.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from arch import arch_model
from datetime import datetime

from mt5_connector import MT5Connector
from constants import SHIFT_PERIODS


# ── Configuration ────────────────────────────────────────────
WIN_SYMBOL = "WINJ26"      # Mini Ibovespa April 2026
WDO_SYMBOL = "WDOK26"      # Mini Dollar  May 2026 (J expired 01-Apr)
LOOKBACK_DAYS = 60          # trading days of daily data
INTRADAY_DAYS = 20          # trading days of 15-min data
BARS_PER_DAY_15M = 28       # ~7h B3 session @ 15-min bars
ANNUALISATION_DAILY = np.sqrt(252)
ANNUALISATION_15M = np.sqrt(252 * BARS_PER_DAY_15M)


# ═════════════════════════════════════════════════════════════
# 1. Kalman Filter for Dynamic Hedge Ratio (spread = WIN - β·WDO - α)
# ═════════════════════════════════════════════════════════════
class KalmanSpreadFilter:
    """
    State-space model:
        WIN_t = α_t + β_t · WDO_t + ε_t     (observation)
        [α_t, β_t]' = [α_{t-1}, β_{t-1}]'   (random walk, process noise Q)

    Returns hedge ratio β, intercept α, spread residuals, and
    innovation-based volatility at each timestep.
    """

    def __init__(self, delta=1e-4, obs_noise=1.0):
        self.delta = delta
        self.R = obs_noise
        self.state = np.array([0.0, -1.0], dtype=np.float64)  # [α, β]
        self.P = np.eye(2) * 1e4
        self.Q = np.eye(2) * delta

        self.alphas = []
        self.betas = []
        self.spreads = []
        self.sqrt_S = []       # innovation std-dev at each step

    def update(self, win_ret, wdo_ret):
        H = np.array([1.0, wdo_ret])
        P_pred = self.P + self.Q

        y_hat = H @ self.state
        innovation = win_ret - y_hat
        S = H @ P_pred @ H + self.R

        K = (P_pred @ H) / S
        self.state = self.state + K * innovation
        self.P = P_pred - np.outer(K, H) @ P_pred

        self.alphas.append(self.state[0])
        self.betas.append(self.state[1])
        self.spreads.append(innovation)
        self.sqrt_S.append(np.sqrt(S))

    def fit(self, win_series, wdo_series):
        self.alphas.clear()
        self.betas.clear()
        self.spreads.clear()
        self.sqrt_S.clear()

        for w, d in zip(win_series, wdo_series):
            self.update(w, d)

        df = pd.DataFrame({
            'alpha': self.alphas,
            'beta': self.betas,
            'spread': self.spreads,
            'innovation_vol': self.sqrt_S,
        })

        # Z-score: normalised innovation (spread / sqrt(S))
        df['z_score'] = df['spread'] / df['innovation_vol']

        return df


# ═════════════════════════════════════════════════════════════
# 2. Data Fetching
# ═════════════════════════════════════════════════════════════
def fetch_pair_data(mt5_conn, timeframe, periods, shift=SHIFT_PERIODS):
    """Fetch aligned price data for WIN and WDO from MT5."""
    df_win = mt5_conn.get_data(WIN_SYMBOL, timeframe, periods, shift)
    df_wdo = mt5_conn.get_data(WDO_SYMBOL, timeframe, periods, shift)

    if df_win is None or df_wdo is None:
        raise RuntimeError(
            f"Could not fetch data for {WIN_SYMBOL} and/or {WDO_SYMBOL}. "
            "Make sure both symbols are in MT5 Market Watch."
        )

    df_w = df_win.set_index('time')[['close']].rename(columns={'close': 'win'})
    df_d = df_wdo.set_index('time')[['close']].rename(columns={'close': 'wdo'})
    merged = df_w.join(df_d, how='inner').dropna()
    merged = merged[(merged['win'] > 0) & (merged['wdo'] > 0)]

    if len(merged) < 20:
        raise RuntimeError(f"Only {len(merged)} aligned bars — need at least 20.")

    return merged


# ═════════════════════════════════════════════════════════════
# 3. GARCH Volatility
# ═════════════════════════════════════════════════════════════
def estimate_garch_vol(spread_returns, freq_label="daily"):
    """
    Fit GARCH(1,1) on the spread return series.
    Returns the model, last conditional vol, and full variance series.
    """
    # arch expects returns scaled by 100 for numerical stability
    y = spread_returns * 100

    model = arch_model(y, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    result = model.fit(disp='off', show_warning=False)

    cond_vol = result.conditional_volatility / 100  # back to decimal
    last_vol = cond_vol.iloc[-1]

    print(f"\n{'='*60}")
    print(f"  GARCH(1,1) — {freq_label} spread returns")
    print(f"{'='*60}")
    print(result.summary().tables[0])
    print(result.summary().tables[1])
    print(f"\n  Latest conditional σ (per-bar): {last_vol:.6f}")

    return result, cond_vol, last_vol


# ═════════════════════════════════════════════════════════════
# 4. Kalman Volatility
# ═════════════════════════════════════════════════════════════
def estimate_kalman_vol(win_returns, wdo_returns, delta=1e-5, obs_noise=1e-4):
    """
    Fit Kalman filter on return-space:  r_WIN = α + β·r_WDO + ε
    Returns KalmanSpreadFilter results DataFrame.
    """
    kf = KalmanSpreadFilter(delta=delta, obs_noise=obs_noise)
    df = kf.fit(win_returns.values, wdo_returns.values)
    df.index = win_returns.index

    print(f"\n{'='*60}")
    print(f"  Kalman Filter — dynamic hedge ratio")
    print(f"{'='*60}")
    print(f"  Latest α  = {kf.state[0]:+.6f}")
    print(f"  Latest β  = {kf.state[1]:+.6f}")
    print(f"  Innovation vol (last) = {df['innovation_vol'].iloc[-1]:.6f}")
    print(f"  Spread residual std   = {df['spread'].std():.6f}")

    return kf, df


# ═════════════════════════════════════════════════════════════
# 5. Plots
# ═════════════════════════════════════════════════════════════
def plot_results(merged, spread_ret, garch_vol, kalman_df, ann_factor, freq_label):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(
        f"Pair Volatility: {WIN_SYMBOL} × {WDO_SYMBOL} — {freq_label}",
        fontsize=14, fontweight='bold'
    )

    # (a) Price series (normalised to 100)
    ax = axes[0]
    win_norm = merged['win'] / merged['win'].iloc[0] * 100
    wdo_norm = merged['wdo'] / merged['wdo'].iloc[0] * 100
    ax.plot(win_norm, label=WIN_SYMBOL, linewidth=1.2)
    ax.plot(wdo_norm, label=WDO_SYMBOL, linewidth=1.2)
    ax.set_ylabel('Normalised Price')
    ax.legend(loc='upper left')
    ax.set_title('(a) Normalised Prices')
    ax.grid(alpha=0.3)

    # (b) Spread log-returns
    ax = axes[1]
    ax.plot(spread_ret, color='steelblue', linewidth=0.7, alpha=0.8)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_ylabel('Spread Return')
    ax.set_title('(b) Spread Log-Returns (r_WIN − β·r_WDO)')
    ax.grid(alpha=0.3)

    # (c) GARCH vs Kalman conditional vol
    ax = axes[2]
    garch_ann = garch_vol * ann_factor
    kalman_ann = kalman_df['innovation_vol'] * ann_factor

    ax.plot(garch_ann.values, label='GARCH(1,1)', linewidth=1.2, color='crimson')
    ax.plot(kalman_ann.values, label='Kalman innovation', linewidth=1.2, color='teal')
    ax.set_ylabel('Annualised σ')
    ax.set_title('(c) Conditional Volatility — GARCH vs Kalman')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # (d) Kalman hedge ratio β
    ax = axes[3]
    ax.plot(kalman_df['beta'].values, color='darkorange', linewidth=1.2)
    ax.set_ylabel('β (hedge ratio)')
    ax.set_title('(d) Kalman Dynamic Hedge Ratio β')
    ax.grid(alpha=0.3)
    ax.set_xlabel('Bar index')

    plt.tight_layout()
    plt.show(block=False)


# ═════════════════════════════════════════════════════════════
# 5b. Intraday Z-Score Analysis & Plot
# ═════════════════════════════════════════════════════════════
ZSCORE_ENTRY = 2.0      # open mean-reversion trade
ZSCORE_EXIT  = 0.5      # close trade (back to fair value)
ZSCORE_STOP  = 3.0      # stop-loss / trend breakout
ZSCORE_LOOKBACK = 20    # rolling window for cumulative spread z


def analyse_intraday_zscore(intra, kf, kalman_df):
    """
    Compute and display Kalman z-score analytics for intraday pair trading.
    Returns the enriched DataFrame.
    """
    df = kalman_df.copy()
    df.index = intra.index[1:]  # skip first bar (no return)
    df['win'] = intra['win'].values[1:]
    df['wdo'] = intra['wdo'].values[1:]

    # Rolling cumulative spread (sum of innovations = price-level spread)
    df['cum_spread'] = df['spread'].cumsum()
    roll_mean = df['cum_spread'].rolling(ZSCORE_LOOKBACK, min_periods=5).mean()
    roll_std  = df['cum_spread'].rolling(ZSCORE_LOOKBACK, min_periods=5).std()
    df['cum_z'] = (df['cum_spread'] - roll_mean) / roll_std

    # Current state
    z_now       = df['z_score'].iloc[-1]
    cum_z_now   = df['cum_z'].iloc[-1]
    spread_now  = df['spread'].iloc[-1]
    beta_now    = df['beta'].iloc[-1]
    win_last    = df['win'].iloc[-1]
    wdo_last    = df['wdo'].iloc[-1]

    # Signal logic
    def signal_from_z(z):
        if np.isnan(z):
            return "WAIT (warming up)"
        az = abs(z)
        if az >= ZSCORE_STOP:
            return "STOP / TREND BREAKOUT"
        elif az >= ZSCORE_ENTRY:
            direction = "SHORT spread" if z > 0 else "LONG spread"
            return f"ENTRY → {direction}"
        elif az <= ZSCORE_EXIT:
            return "EXIT / FLAT (near fair value)"
        else:
            return "HOLD (no action)"

    sig_instant = signal_from_z(z_now)
    sig_cumul   = signal_from_z(cum_z_now)

    # Print report
    print(f"\n{'='*70}")
    print(f"  KALMAN Z-SCORE — INTRADAY PAIR TRADE SIGNALS")
    print(f"  {WIN_SYMBOL} × {WDO_SYMBOL}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    print(f"  {WIN_SYMBOL} = {win_last:,.0f}    {WDO_SYMBOL} = {wdo_last:,.2f}")
    print(f"  Kalman β = {beta_now:+.6f}   (1 WIN ≈ {abs(beta_now):.2f} WDO hedge)")
    print(f"{'─'*70}")
    print(f"  Instantaneous z-score (innovation / √S):  {z_now:+.4f}")
    print(f"  Cumulative z-score   (rolling {ZSCORE_LOOKBACK}-bar):    "
          f"{cum_z_now:+.4f}" if not np.isnan(cum_z_now) else "  N/A (warming)")
    print(f"{'─'*70}")
    print(f"  Signal (instant z):  {sig_instant}")
    print(f"  Signal (cumul.  z):  {sig_cumul}")
    print(f"{'─'*70}")
    print(f"  Thresholds:  Entry ±{ZSCORE_ENTRY}  |  Exit ±{ZSCORE_EXIT}  |  Stop ±{ZSCORE_STOP}")
    print()

    # Last N bars table
    N_TAIL = 15
    tail = df[['win', 'wdo', 'beta', 'spread', 'z_score', 'cum_z']].tail(N_TAIL)
    print(f"  Last {N_TAIL} bars:")
    header = f"  {'Time':>19s}  {'WIN':>9s}  {'WDO':>9s}  {'β':>8s}  {'Spread':>10s}  {'z_inst':>8s}  {'z_cum':>8s}"
    print(header)
    print(f"  {'─'*len(header)}")
    for ts, row in tail.iterrows():
        ts_str = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)
        cum_str = f"{row['cum_z']:+8.3f}" if not np.isnan(row['cum_z']) else f"{'N/A':>8s}"
        print(f"  {ts_str:>19s}  {row['win']:>9,.0f}  {row['wdo']:>9,.2f}  "
              f"{row['beta']:>+8.4f}  {row['spread']:>+10.6f}  "
              f"{row['z_score']:>+8.3f}  {cum_str}")
    print()

    return df


def plot_zscore(df_z, freq_label="Intraday (15-min)"):
    """Plot z-score chart with entry/exit bands."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Kalman Z-Score: {WIN_SYMBOL} × {WDO_SYMBOL} — {freq_label}",
        fontsize=14, fontweight='bold'
    )

    # (a) Cumulative spread
    ax = axes[0]
    ax.plot(df_z.index, df_z['cum_spread'], color='steelblue', linewidth=1.0)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_ylabel('Cumulative Spread')
    ax.set_title('(a) Kalman Cumulative Spread (Σ innovations)')
    ax.grid(alpha=0.3)

    # (b) Instantaneous z-score
    ax = axes[1]
    ax.plot(df_z.index, df_z['z_score'], color='navy', linewidth=0.8, alpha=0.9)
    ax.axhline(0, color='grey', linewidth=0.5)
    for lvl, col, ls in [(ZSCORE_ENTRY, 'green', '--'), (-ZSCORE_ENTRY, 'green', '--'),
                          (ZSCORE_STOP, 'red', ':'), (-ZSCORE_STOP, 'red', ':'),
                          (ZSCORE_EXIT, 'orange', '-.'), (-ZSCORE_EXIT, 'orange', '-.')]:
        ax.axhline(lvl, color=col, linewidth=0.8, linestyle=ls, alpha=0.7)
    ax.fill_between(df_z.index, -ZSCORE_ENTRY, ZSCORE_ENTRY, alpha=0.05, color='green')
    ax.fill_between(df_z.index, ZSCORE_ENTRY, ZSCORE_STOP, alpha=0.08, color='yellow')
    ax.fill_between(df_z.index, -ZSCORE_STOP, -ZSCORE_ENTRY, alpha=0.08, color='yellow')
    ax.set_ylabel('z-score')
    ax.set_title('(b) Instantaneous Z-Score (innovation / √S)')
    ax.legend(['z-score', f'Entry ±{ZSCORE_ENTRY}', '', f'Stop ±{ZSCORE_STOP}', '',
               f'Exit ±{ZSCORE_EXIT}'], loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Cumulative z-score (rolling)
    ax = axes[2]
    ax.plot(df_z.index, df_z['cum_z'], color='darkviolet', linewidth=1.0)
    ax.axhline(0, color='grey', linewidth=0.5)
    for lvl, col, ls in [(ZSCORE_ENTRY, 'green', '--'), (-ZSCORE_ENTRY, 'green', '--'),
                          (ZSCORE_STOP, 'red', ':'), (-ZSCORE_STOP, 'red', ':')]:
        ax.axhline(lvl, color=col, linewidth=0.8, linestyle=ls, alpha=0.7)
    ax.set_ylabel('z-score')
    ax.set_title(f'(c) Cumulative Z-Score (rolling {ZSCORE_LOOKBACK}-bar)')
    ax.set_xlabel('Time')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)


# ═════════════════════════════════════════════════════════════
# 6. Summary Table
# ═════════════════════════════════════════════════════════════
def print_summary(garch_daily_vol, kalman_daily_vol, garch_intra_vol,
                  kalman_intra_vol, kalman_beta_d, kalman_beta_i):

    print(f"\n{'='*70}")
    print(f"  PAIR VOLATILITY SUMMARY — {WIN_SYMBOL} × {WDO_SYMBOL}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    row = "{:<30s} {:>18s} {:>18s}"
    print(row.format("Metric", "Daily (D1)", "Intraday (M15)"))
    print("-" * 70)

    def fmt(v, ann):
        return f"{v:.6f} ({v*ann*100:.2f}% ann)"

    print(row.format("GARCH(1,1) σ per-bar",
                      f"{garch_daily_vol:.6f}",
                      f"{garch_intra_vol:.6f}"))
    print(row.format("GARCH(1,1) σ annualised",
                      f"{garch_daily_vol*ANNUALISATION_DAILY*100:.2f}%",
                      f"{garch_intra_vol*ANNUALISATION_15M*100:.2f}%"))
    print(row.format("Kalman innov. σ per-bar",
                      f"{kalman_daily_vol:.6f}",
                      f"{kalman_intra_vol:.6f}"))
    print(row.format("Kalman innov. σ annualised",
                      f"{kalman_daily_vol*ANNUALISATION_DAILY*100:.2f}%",
                      f"{kalman_intra_vol*ANNUALISATION_15M*100:.2f}%"))
    print(row.format("Kalman hedge ratio β",
                      f"{kalman_beta_d:+.6f}",
                      f"{kalman_beta_i:+.6f}"))
    print("=" * 70)


# ═════════════════════════════════════════════════════════════
# 7. Main
# ═════════════════════════════════════════════════════════════
def main():
    mt5 = MT5Connector()

    # ── Daily analysis ──────────────────────────────────────
    print("\n" + "▓" * 60)
    print("  DAILY (D1) ANALYSIS")
    print("▓" * 60)

    daily = fetch_pair_data(mt5, mt5.TIMEFRAME_D1, LOOKBACK_DAYS)
    print(f"  Fetched {len(daily)} daily bars  "
          f"({daily.index[0].date()} → {daily.index[-1].date()})")
    print(f"  {WIN_SYMBOL} last: {daily['win'].iloc[-1]:,.0f}  "
          f"{WDO_SYMBOL} last: {daily['wdo'].iloc[-1]:,.2f}")

    # Log returns
    daily_ret = np.log(daily / daily.shift(1)).dropna()

    # Simple spread return (before Kalman, use OLS β for GARCH input)
    ols_beta_d = np.polyfit(daily_ret['wdo'], daily_ret['win'], 1)[0]
    spread_ret_d = daily_ret['win'] - ols_beta_d * daily_ret['wdo']
    print(f"  OLS hedge ratio β = {ols_beta_d:+.6f}")

    # GARCH
    garch_res_d, garch_vol_d, garch_last_d = estimate_garch_vol(
        spread_ret_d, freq_label="daily (D1)")

    # Kalman
    kf_d, kalman_df_d = estimate_kalman_vol(
        daily_ret['win'], daily_ret['wdo'],
        delta=1e-5, obs_noise=1e-4)

    kalman_last_d = kalman_df_d['innovation_vol'].iloc[-1]

    plot_results(daily, spread_ret_d, garch_vol_d, kalman_df_d,
                 ANNUALISATION_DAILY, "Daily (D1)")

    # ── Intraday analysis ───────────────────────────────────
    print("\n" + "▓" * 60)
    print("  INTRADAY (15-min) ANALYSIS")
    print("▓" * 60)

    n_bars_intra = INTRADAY_DAYS * BARS_PER_DAY_15M + 10
    intra = fetch_pair_data(mt5, mt5.TIMEFRAME_M15, n_bars_intra)
    print(f"  Fetched {len(intra)} 15-min bars  "
          f"({intra.index[0]} → {intra.index[-1]})")

    intra_ret = np.log(intra / intra.shift(1)).dropna()

    ols_beta_i = np.polyfit(intra_ret['wdo'], intra_ret['win'], 1)[0]
    spread_ret_i = intra_ret['win'] - ols_beta_i * intra_ret['wdo']
    print(f"  OLS hedge ratio β = {ols_beta_i:+.6f}")

    # GARCH
    garch_res_i, garch_vol_i, garch_last_i = estimate_garch_vol(
        spread_ret_i, freq_label="intraday (15-min)")

    # Kalman
    kf_i, kalman_df_i = estimate_kalman_vol(
        intra_ret['win'], intra_ret['wdo'],
        delta=1e-6, obs_noise=1e-5)

    kalman_last_i = kalman_df_i['innovation_vol'].iloc[-1]

    plot_results(intra, spread_ret_i, garch_vol_i, kalman_df_i,
                 ANNUALISATION_15M, "Intraday (15-min)")

    # ── Intraday z-score analysis ───────────────────────────
    df_z = analyse_intraday_zscore(intra, kf_i, kalman_df_i)
    plot_zscore(df_z)

    # ── Combined summary ────────────────────────────────────
    print_summary(
        garch_daily_vol=garch_last_d,
        kalman_daily_vol=kalman_last_d,
        garch_intra_vol=garch_last_i,
        kalman_intra_vol=kalman_last_i,
        kalman_beta_d=kf_d.state[1],
        kalman_beta_i=kf_i.state[1],
    )

    plt.show()


if __name__ == "__main__":
    main()

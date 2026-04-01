# -*- coding: utf-8 -*-
"""
Kalman Filter — $IND ↔ BOVA11 Price Mapper & Hedge Sizing
----------------------------------------------------------
Provides:
  - KalmanPriceMapper      : Kalman-based dynamic regression IND = α + β·BOVA11
  - build_ind_bova11_mapper: fits the mapper from MT5 historical data
  - calculate_hedge_options : notional-based hedge sizing
  - calculate_delta_neutral_hedge : delta-neutral hedge from live option chain

"""

import numpy as np
import pandas as pd

from constants import PERIODS, SHIFT_PERIODS


# ============================================================
# Kalman Filter — $IND ↔ BOVA11 Price Mapper
# ============================================================
class KalmanPriceMapper:
    """
    Uses a Kalman Filter to dynamically estimate the linear relationship
    between $IND (Bovespa index futures) and BOVA11 (Bovespa ETF):

        P_IND = alpha + beta * P_BOVA11

    After fitting on historical data, converts BOVA11 prices (e.g. GEX
    strike levels) to their $IND equivalents and vice-versa.

    State vector: [alpha, beta]
    Observation:  P_IND_t = [1, P_BOVA11_t] · [alpha_t, beta_t]' + noise
    """

    def __init__(self,
                 delta: float = 1e-4,
                 observation_noise: float = 100.0,
                 initial_alpha: float = 0.0,
                 initial_beta: float = 1000.0,
                 initial_variance: float = 1e4):
        """
        Parameters
        ----------
        delta : float
            Process noise scalar — controls how fast alpha/beta can drift.
        observation_noise : float
            Measurement noise variance (R). For IND points ~130 000, a
            value around 100–1000 is reasonable.
        initial_alpha : float
            Starting intercept guess.
        initial_beta : float
            Starting slope guess (IND / BOVA11 ≈ 1000).
        initial_variance : float
            Diagonal of the initial state covariance P_0.
        """
        self.delta = delta
        self.R = observation_noise

        # State: [alpha, beta]
        self.state = np.array([initial_alpha, initial_beta], dtype=np.float64)
        self.P = np.eye(2) * initial_variance        # state covariance
        self.Q = np.eye(2) * delta                    # process noise

        # History
        self.alphas: list = []
        self.betas: list = []

    # ── core update ──────────────────────────────────────────
    def update(self, ind_price: float, bova11_price: float):
        """Single-step Kalman update with new price pair."""
        H = np.array([1.0, bova11_price])              # observation vector

        # Predict
        P_pred = self.P + self.Q

        # Innovation
        y_hat = H @ self.state
        innovation = ind_price - y_hat
        S = H @ P_pred @ H + self.R                     # innovation variance

        # Kalman gain
        K = (P_pred @ H) / S                             # (2,)

        # Update
        self.state = self.state + K * innovation
        self.P = P_pred - np.outer(K, H) @ P_pred

        self.alphas.append(self.state[0])
        self.betas.append(self.state[1])

    # ── batch fit ────────────────────────────────────────────
    def fit(self, ind_prices: np.ndarray, bova11_prices: np.ndarray) -> pd.DataFrame:
        """
        Run the filter over aligned historical arrays.

        Returns a DataFrame with columns:
            ind, bova11, alpha, beta, ind_estimated, residual
        """
        assert len(ind_prices) == len(bova11_prices), "Series must be same length"

        self.alphas.clear()
        self.betas.clear()

        for ind_p, bova_p in zip(ind_prices, bova11_prices):
            self.update(ind_p, bova_p)

        alphas = np.array(self.alphas)
        betas = np.array(self.betas)
        estimated = alphas + betas * bova11_prices

        return pd.DataFrame({
            'ind': ind_prices,
            'bova11': bova11_prices,
            'alpha': alphas,
            'beta': betas,
            'ind_estimated': estimated,
            'residual': ind_prices - estimated,
        })

    # ── conversion helpers ───────────────────────────────────
    @property
    def alpha(self) -> float:
        return self.state[0]

    @property
    def beta(self) -> float:
        return self.state[1]


    def bova11_to_ind(self, bova11_price: float, log_input: bool = False) -> float:
        """Convert a BOVA11 price to the corresponding $IND price.
        If the Kalman filter was fit on log prices, exponentiate the result to get the actual price.
        Set log_input=True if passing a log price, otherwise pass the actual price.
        """
        if log_input:
            ind_log = self.alpha + self.beta * bova11_price
            return np.exp(ind_log)
        else:
            # If input is price, convert to log, apply, then exponentiate
            ind_log = self.alpha + self.beta * np.log(bova11_price)
            return np.exp(ind_log)

    def ind_to_bova11(self, ind_price: float, log_input: bool = False) -> float:
        """Convert an $IND price to the corresponding BOVA11 price.
        If the Kalman filter was fit on log prices, exponentiate the result to get the actual price.
        Set log_input=True if passing a log price, otherwise pass the actual price.
        """
        if self.beta == 0:
            raise ValueError("beta is zero — filter not fitted yet")
        if log_input:
            bova11_log = (ind_price - self.alpha) / self.beta
            return np.exp(bova11_log)
        else:
            # If input is price, convert to log, apply, then exponentiate
            bova11_log = (np.log(ind_price) - self.alpha) / self.beta
            return np.exp(bova11_log)

    def convert_strikes(self, strikes: np.ndarray) -> np.ndarray:
        """Convert an array of BOVA11 option strikes to $IND equivalents."""
        return self.alpha + self.beta * np.asarray(strikes, dtype=np.float64)


# ============================================================
# Builder — fetch from MT5 and fit
# ============================================================
def build_ind_bova11_mapper(mt5_conn,
                            ind_symbol: str = "WIN$N",
                            bova11_symbol: str = "BOVA11",
                            periods: int = PERIODS,
                            delta: float = 1e-4,
                            observation_noise: float = 100.0) -> KalmanPriceMapper:
    """
    Fetch historical daily close prices from MT5 for $IND and BOVA11,
    fit a KalmanPriceMapper, and return it ready for conversions.

    Parameters
    ----------
    mt5_conn : MT5Connector
        An already-initialised MT5 connector.
    ind_symbol : str
        MT5 symbol for the continuous index future (e.g. "WIN$N", "IND$").
    bova11_symbol : str
        MT5 symbol for the ETF.
    periods : int
        Number of historical bars (daily) to use for calibration.
    delta : float
        Kalman process noise.
    observation_noise : float
        Kalman measurement noise.

    Returns
    -------
    KalmanPriceMapper
        Fitted mapper. Call .bova11_to_ind(price) or .ind_to_bova11(price).
    """
    df_ind = mt5_conn.get_data(ind_symbol, mt5_conn.TIMEFRAME_D1, periods, SHIFT_PERIODS)
    df_bova = mt5_conn.get_data(bova11_symbol, mt5_conn.TIMEFRAME_D1, periods, SHIFT_PERIODS)

    if df_ind is None or df_bova is None:
        raise RuntimeError(
            f"Could not fetch data for {ind_symbol} and/or {bova11_symbol}. "
            "Make sure both symbols are available in MT5 Market Watch."
        )

    # Align on date
    df_ind = df_ind.set_index('time')[['close']].rename(columns={'close': 'ind'})
    df_bova = df_bova.set_index('time')[['close']].rename(columns={'close': 'bova11'})
    merged = df_ind.join(df_bova, how='inner').dropna()

    if len(merged) < 5:
        raise RuntimeError(
            f"Only {len(merged)} overlapping bars -- need at least 5 for "
            "a reliable calibration."
        )

    # Calculate log prices
    merged['ind_log'] = np.log(merged['ind'])
    merged['bova11_log'] = np.log(merged['bova11'])
    merged = merged.dropna(subset=['ind_log', 'bova11_log'])

    # Estimate sensible initial beta from OLS on last 60 log prices
    lookback = min(60, len(merged))
    # OLS: y = alpha + beta * x, so x = bova11_log, y = ind_log
    ols_coeffs = np.polyfit(
        merged['bova11_log'].values[-lookback:],
        merged['ind_log'].values[-lookback:], 1
    )
    ols_beta = ols_coeffs[0]

    mapper = KalmanPriceMapper(
        delta=delta,
        observation_noise=observation_noise,
        initial_beta=ols_beta,
    )
    results = mapper.fit(
        merged['ind_log'].values,
        merged['bova11_log'].values
    )

    print(f"[KalmanPriceMapper] Fitted on {len(merged)} daily log prices")
    print(f"  Latest alpha = {mapper.alpha:,.6f}  beta = {mapper.beta:,.6f}")
    print(f"  Residual std = {results['residual'].std():,.6f} (log price units)")
    print(f"  Example: BOVA11 log {merged['bova11_log'].iloc[-1]:.6f} -> "
          f"$IND log {mapper.bova11_to_ind(merged['bova11_log'].iloc[-1]):,.6f} "
          f"(actual {merged['ind_log'].iloc[-1]:,.6f})")

    return mapper


# ============================================================
# Intraday 15-min Builder -- auto-select best lookback
# ============================================================
# B3 regular session: ~7 hours = 28 bars of 15 min per day
BARS_PER_DAY_15M = 28


def _build_candidate_days(max_days: int) -> list:
    """Generate candidate lookback periods up to max_days.
    
    Always includes 1, 2, 3 and then adds 5, 10, 15, 20 if they fit.
    This ensures we always test short windows plus whatever the user sets.
    """
    base = [1, 2, 3, 5, 10, 15, 20]
    # Include max_days itself if not already in the list
    candidates = sorted(set([d for d in base if d <= max_days] + [max_days]))
    return candidates


def build_ind_bova11_mapper_intraday(
    mt5_conn,
    ind_symbol: str = "WIN$N",
    bova11_symbol: str = "BOVA11",
    max_days: int = PERIODS,
    delta: float = 1e-3,
    observation_noise: float = 10.0,
) -> KalmanPriceMapper:
    """
    Fit a KalmanPriceMapper on 15-min bars, automatically selecting the
    lookback period that minimises out-of-sample residual error.

    The candidate lookback periods are generated from PERIODS (constants.py).
    Change PERIODS to control the max lookback tested.

    Strategy
    --------
    For each candidate lookback (up to max_days trading days) we:
      1. Fetch 15-min bars and align timestamps.
      2. Split 80% train / 20% test.
      3. Fit the Kalman filter on train.
      4. Measure mean-absolute-error (MAE) on test (in index points).
      5. Pick the lookback with lowest test MAE.

    Then re-fit the winner on the full period and return the mapper.

    Parameters
    ----------
    max_days : int
        Maximum lookback in trading days to test. Driven by PERIODS in
        constants.py. Default candidates: 1,2,3 and up to max_days.

    Returns
    -------
    KalmanPriceMapper
        Fitted on 15-min bars, ready for bova11_to_ind / ind_to_bova11.
    """
    candidate_days = _build_candidate_days(max_days)
    
    best_mae = np.inf
    best_days = candidate_days[0]
    results_table = []

    for days in candidate_days:
        n_bars = days * BARS_PER_DAY_15M + 10  # small buffer

        df_ind = mt5_conn.get_data(ind_symbol, mt5_conn.TIMEFRAME_M15,
                                   n_bars, SHIFT_PERIODS)
        df_bova = mt5_conn.get_data(bova11_symbol, mt5_conn.TIMEFRAME_M15,
                                    n_bars, SHIFT_PERIODS)

        if df_ind is None or df_bova is None:
            results_table.append((days, n_bars, 0, np.nan, "no data"))
            continue

        # Align on timestamp
        df_i = df_ind.set_index('time')[['close']].rename(columns={'close': 'ind'})
        df_b = df_bova.set_index('time')[['close']].rename(columns={'close': 'bova11'})
        merged = df_i.join(df_b, how='inner').dropna()
        merged = merged[(merged['ind'] > 0) & (merged['bova11'] > 0)]

        if len(merged) < 10:
            results_table.append((days, n_bars, len(merged), np.nan, "too few bars"))
            continue

        # Log prices
        ind_log = np.log(merged['ind'].values)
        bova_log = np.log(merged['bova11'].values)

        # Train/test split (80/20)
        split = int(len(merged) * 0.8)
        train_ind, test_ind = ind_log[:split], ind_log[split:]
        train_bova, test_bova = bova_log[:split], bova_log[split:]

        # OLS seed
        lookback = min(60, len(train_ind))
        ols_coeffs = np.polyfit(train_bova[-lookback:], train_ind[-lookback:], 1)

        # Fit on train
        mapper = KalmanPriceMapper(
            delta=delta,
            observation_noise=observation_noise,
            initial_beta=ols_coeffs[0],
        )
        mapper.fit(train_ind, train_bova)

        # Evaluate on test -- step through with updates
        errors = []
        for t_ind, t_bova in zip(test_ind, test_bova):
            predicted = mapper.alpha + mapper.beta * t_bova
            error_pts = abs(np.exp(t_ind) - np.exp(predicted))
            errors.append(error_pts)
            mapper.update(t_ind, t_bova)

        mae = np.mean(errors)
        results_table.append((days, n_bars, len(merged), mae, "ok"))

        if mae < best_mae:
            best_mae = mae
            best_days = days

    # Print comparison table
    print(f"\n[KalmanPriceMapper] Intraday 15-min -- lookback selection:")
    print(f"  {'Days':>5s}  {'Bars':>6s}  {'Aligned':>7s}  {'MAE (pts)':>10s}  Status")
    print(f"  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}")
    for days, n_bars, n_aligned, mae, status in results_table:
        mae_str = f"{mae:>10.1f}" if np.isfinite(mae) else f"{'N/A':>10s}"
        marker = " <-- BEST" if days == best_days and np.isfinite(mae) else ""
        print(f"  {days:>5d}  {n_bars:>6d}  {n_aligned:>7d}  {mae_str}  {status}{marker}")

    # Re-fit best period on ALL data
    n_bars = best_days * BARS_PER_DAY_15M + 10
    df_ind = mt5_conn.get_data(ind_symbol, mt5_conn.TIMEFRAME_M15,
                               n_bars, SHIFT_PERIODS)
    df_bova = mt5_conn.get_data(bova11_symbol, mt5_conn.TIMEFRAME_M15,
                                n_bars, SHIFT_PERIODS)

    df_i = df_ind.set_index('time')[['close']].rename(columns={'close': 'ind'})
    df_b = df_bova.set_index('time')[['close']].rename(columns={'close': 'bova11'})
    merged = df_i.join(df_b, how='inner').dropna()
    merged = merged[(merged['ind'] > 0) & (merged['bova11'] > 0)]

    ind_log = np.log(merged['ind'].values)
    bova_log = np.log(merged['bova11'].values)

    lookback = min(60, len(merged))
    ols_coeffs = np.polyfit(bova_log[-lookback:], ind_log[-lookback:], 1)

    final_mapper = KalmanPriceMapper(
        delta=delta,
        observation_noise=observation_noise,
        initial_beta=ols_coeffs[0],
    )
    results_df = final_mapper.fit(ind_log, bova_log)

    print(f"\n  Winner: {best_days} days ({len(merged)} bars of 15-min)")
    print(f"  MAE:    {best_mae:.1f} index points")
    print(f"  alpha = {final_mapper.alpha:,.6f}  beta = {final_mapper.beta:,.6f}")
    print(f"  Residual std = {results_df['residual'].std():,.6f} (log units)")

    return final_mapper


# ============================================================
# WIN <-> BOVA11 Options Hedge Sizing
# ============================================================
WIN_POINT_VALUE = 0.20  # BRL per index point for WIN mini contracts


def calculate_hedge_options(mapper: KalmanPriceMapper,
                            win_contracts: int = 1,
                            option_delta: float = 0.50,
                            bova11_price: float = None,
                            ind_price: float = None) -> dict:
    """
    Calculate the number of BOVA11 options needed to hedge WIN mini futures.

    The notional of 1 WIN contract is:
        WIN_notional = WIN_POINT_VALUE × IND_price

    Each BOVA11 option covers 1 share, and its directional exposure is:
        option_exposure = BOVA11_price × delta

    To translate between the two markets we use the Kalman β so that
    a 1-point move in BOVA11 ≈ β points in IND:
        hedge_exposure_per_option = option_exposure × β × WIN_POINT_VALUE

    Number of options:
        N = (win_contracts × WIN_notional) / hedge_exposure_per_option

    Parameters
    ----------
    mapper : KalmanPriceMapper
        Fitted Kalman mapper (provides α, β and price conversions).
    win_contracts : int
        Number of WIN mini contracts to hedge.
    option_delta : float
        Delta of the BOVA11 option chosen for the hedge (e.g. 0.50 for ATM,
        -0.30 for OTM put — pass the absolute value).
    bova11_price : float, optional
        Current BOVA11 price. If None, derived from ind_price via the mapper.
    ind_price : float, optional
        Current $IND price. If None, derived from bova11_price via the mapper.

    Returns
    -------
    dict with keys:
        n_options       — number of BOVA11 options (rounded up)
        win_notional    — notional of the WIN position in BRL
        option_exposure — directional BRL exposure per option
        beta            — Kalman β used
        ind_price       — $IND price used
        bova11_price    — BOVA11 price used
    """
    if bova11_price is None and ind_price is None:
        raise ValueError("Provide at least one of bova11_price or ind_price")

    if ind_price is None:
        ind_price = mapper.bova11_to_ind(bova11_price)
    if bova11_price is None:
        bova11_price = mapper.ind_to_bova11(ind_price)

    abs_delta = abs(option_delta)
    beta = mapper.beta

    win_notional = win_contracts * WIN_POINT_VALUE * ind_price
    option_exposure = abs_delta * bova11_price * beta * WIN_POINT_VALUE
    n_options = int(np.ceil(win_notional / option_exposure))

    return {
        'n_options': n_options,
        'win_notional': win_notional,
        'option_exposure': option_exposure,
        'beta': beta,
        'ind_price': ind_price,
        'bova11_price': bova11_price,
    }


def calculate_delta_neutral_hedge(mapper: KalmanPriceMapper,
                                  options_df: pd.DataFrame,
                                  spot: float,
                                  win_contracts: int = 1,
                                  side: str = 'put',
                                  max_dte: int = 60) -> dict:
    """
    Estimate the number of BOVA11 options required to make a WIN futures
    position delta-neutral, picking the best candidate from the live chain.

    A long WIN contract has a delta of +1 in index-point terms.  In BRL:
        futures_delta_brl = win_contracts × WIN_POINT_VALUE × β

    Each BOVA11 option has BS delta Δ_opt, so its BRL delta is:
        option_delta_brl  = Δ_opt × β × WIN_POINT_VALUE

    To neutralise:
        N = -futures_delta_brl / option_delta_brl
          = -win_contracts / Δ_opt          (β cancels out)

    For a long WIN hedged with puts (Δ < 0)  → N > 0  (buy puts)
    For a short WIN hedged with calls (Δ > 0) → N > 0  (buy calls)

    The function selects the option closest to ATM within the requested
    side and DTE window, then returns the quantity and full details.

    Parameters
    ----------
    mapper : KalmanPriceMapper
        Fitted Kalman mapper (used for $IND price conversion in output).
    options_df : pd.DataFrame
        Options chain from load_b3_options_data() — must contain columns
        'Strike', 'Delta', 'Tipo', 'DTE', 'Ticker', 'IV'.
    spot : float
        Current BOVA11 spot price.
    win_contracts : int
        Number of WIN contracts to hedge (positive = long futures).
    side : str
        'put' to hedge a long futures position, 'call' for short futures.
    max_dte : int
        Maximum days-to-expiry filter for candidate options.

    Returns
    -------
    dict with keys:
        n_options         — number of options to trade (rounded up)
        ticker            — ticker of the selected option
        strike            — strike of the selected option
        option_delta      — BS delta of the selected option
        iv                — implied vol of the selected option
        dte               — days to expiry
        net_delta         — residual delta after hedge (ideally ~0)
        ind_spot          — $IND price corresponding to BOVA11 spot
        ind_strike        — $IND price corresponding to option strike
    """
    tipo = 'PUT' if side.lower() == 'put' else 'CALL'
    candidates = options_df[
        (options_df['Tipo'].str.upper() == tipo) &
        (options_df['DTE'] <= max_dte) &
        (options_df['DTE'] > 0) &
        (options_df['Delta'].abs() > 0.01)
    ].copy()

    if candidates.empty:
        raise ValueError(f"No {tipo} options found with DTE ≤ {max_dte}")

    # Pick strike closest to spot (ATM)
    candidates['dist_to_atm'] = (candidates['Strike'] - spot).abs()
    best = candidates.sort_values('dist_to_atm').iloc[0]

    opt_delta = best['Delta']  # negative for puts, positive for calls

    # N = -futures_delta / option_delta
    # For long WIN (delta=+1 per contract) hedged with puts (delta<0) → N>0
    n_options = int(np.ceil(abs(win_contracts / opt_delta)))

    # Residual delta after hedge
    net_delta = win_contracts + n_options * opt_delta

    return {
        'n_options': n_options,
        'ticker': best['Ticker'],
        'strike': best['Strike'],
        'option_delta': opt_delta,
        'iv': best['IV'],
        'dte': int(best['DTE']),
        'net_delta': net_delta,
        'ind_spot': mapper.bova11_to_ind(spot),
        'ind_strike': mapper.bova11_to_ind(best['Strike']),
    }


# ============================================================
# Kalman Parameter Grid Search Utility
# ============================================================
def kalman_grid_search(ind_prices, bova11_prices, 
                      delta_grid=None, obs_noise_grid=None, 
                      initial_alpha=0.0, initial_beta=None, initial_variance=1e4):
    """
    Grid search for best Kalman filter noise parameters on given price series.
    Returns a DataFrame with parameter combinations and their residual std.
    """
    if delta_grid is None:
        delta_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    if obs_noise_grid is None:
        obs_noise_grid = [10, 50, 100, 250, 500, 1000]
    if initial_beta is None:
        # OLS estimate for initial beta
        initial_beta = np.polyfit(bova11_prices, ind_prices, 1)[0]

    results = []
    for delta in delta_grid:
        for obs_noise in obs_noise_grid:
            kf = KalmanPriceMapper(
                delta=delta,
                observation_noise=obs_noise,
                initial_alpha=initial_alpha,
                initial_beta=initial_beta,
                initial_variance=initial_variance,
            )
            df = kf.fit(ind_prices, bova11_prices)
            resid_std = df['residual'].std()
            results.append({
                'delta': delta,
                'observation_noise': obs_noise,
                'residual_std': resid_std,
            })
    return pd.DataFrame(results).sort_values('residual_std')

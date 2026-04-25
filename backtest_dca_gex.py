# -*- coding: utf-8 -*-
"""
DCA GEX Strategy — Historical Backtest (Last Week)
====================================================
Uses cached B3 COTAHIST options data + MT5 intraday WIN bars
to replay the strategy with real prices and computed GEX walls.

Flow per trading day:
  1. Load B3 options chain for that date → compute GEX walls
  2. Build Kalman mapper for BOVA11↔WIN on that day's bars
  3. Fetch 5-min WIN bars for intraday replay
  4. Simulate the DCA GEX monitor logic bar-by-bar
"""
import os
import sys
import asyncio
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure local imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from constants import (
    GEX_MARGIN_FREE_PCT, GEX_SL_RISK_PCT, GEX_TRAILING_ACTIVATION_PCT,
    GEX_DCA_LOSS_STEP_PCT, GEX_DCA_MAX_ORDERS, GEX_ORDER_VOLUME,
    GEX_WALL_PROXIMITY_PCT, GEX_MIN_SIGNAL_STRENGTH,
    GEX_MIN_SL_POINTS, GEX_TRAILING_DISTANCE_FACTOR,
    GEX_MAX_DAILY_LOSS_PCT, GEX_TP_AT_OPPOSITE_WALL,
    GEX_TRADE_WINDOW_START, GEX_TRADE_WINDOW_END,
    GEX_REQUIRE_5M_CONFIRMATION, GEX_CONFIRMATION_MINUTES,
    GEX_NEUTRAL_ONLY, GEX_NEUTRAL_MAX_FLIP_DISTANCE_PCT,
    ASSET_SYMBOL,
)
from gex_utils import compute_weekly_walls, find_gamma_flip, generate_gex_trade_signals
from gex_zones import select_significant_zones as _select_significant_zones
from bs_greeks import bs_price


def _is_neutral_setup(spot, gamma_flip, call_wall, put_wall, max_flip_distance_pct):
    """Neutral setup: spot between walls and close to gamma flip."""
    if not np.isfinite(spot) or not np.isfinite(gamma_flip) or gamma_flip == 0:
        return False
    if np.isfinite(call_wall) and np.isfinite(put_wall):
        lo = min(call_wall, put_wall)
        hi = max(call_wall, put_wall)
        if not (lo <= spot <= hi):
            return False
    flip_dist = abs((spot - gamma_flip) / gamma_flip)
    return flip_dist <= max_flip_distance_pct

# ── WIN contract specs ───────────────────────────────────────────────
TICK_SIZE  = 5
PNL_PER_POINT = 0.20  # R$ per point per contract

# ── DCA parameters (must match main.py) ─────────────────────────────
FIB_SEQ = [1, 1, 2, 3, 5, 8, 13, 21]
FIB_TOTAL = 1 + sum(FIB_SEQ[:GEX_DCA_MAX_ORDERS])

# ── Simulation assumptions ──────────────────────────────────────────
FREE_MARGIN = 10_000.0
MARGIN_PER_LOT = 100.0
MARGIN_BUDGET = FREE_MARGIN * GEX_MARGIN_FREE_PCT


def align_tick(price):
    return round(round(price / TICK_SIZE) * TICK_SIZE)


# ═══════════════════════════════════════════════════════════════════════
#  0DTE OPTION STRATEGY RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════

def _load_0dte_chain(asset, spot, date_str):
    """
    Load the B3 options chain for *asset* on *date_str* and return
    options expiring within 0-1 business days (0DTE on Fridays,
    1DTE on Thursdays — both valid for short-dated strategies).

    Unlike ``load_b3_options_data`` which computes DTE from today,
    this computes DTE relative to the backtest date itself so that
    historical near-expiry options are correctly identified.
    """
    from get_b3_data import fetch_b3_historical_file
    from bs_greeks import bs_gamma as _bs_gamma, bs_delta as _bs_delta, implied_vol as _iv
    from di1_rate_curve import get_rate_for_date, FALLBACK_RATE

    raw = fetch_b3_historical_file(date_str)
    if raw.empty:
        return pd.DataFrame()

    prefix = asset[:4].upper()
    call_letters = set('ABCDEFGHIJKL')
    put_letters  = set('MNOPQRSTUVWX')

    opts = raw[raw['ticker'].str.startswith(prefix, na=False)].copy()
    if opts.empty:
        return pd.DataFrame()

    def classify(ticker):
        if len(ticker) > 4:
            l = ticker[4].upper()
            if l in call_letters: return 'CALL'
            if l in put_letters:  return 'PUT'
        return None

    opts['Tipo'] = opts['ticker'].apply(classify)
    opts = opts.dropna(subset=['Tipo'])

    ref_date = datetime.strptime(date_str, '%Y-%m-%d').date()

    def parse_exp(exp_str):
        try:
            exp = datetime.strptime(str(exp_str).strip(), '%Y%m%d')
            dte = max(int(np.busday_count(ref_date, exp.date())), 0)
            return dte, exp
        except (ValueError, TypeError):
            return 999, None

    parsed = opts['expiration'].apply(parse_exp)
    opts['DTE'] = parsed.apply(lambda x: x[0])
    opts['Expiration'] = parsed.apply(lambda x: x[1])

    # Keep 0DTE (Friday = expiry day) or 1DTE (Thursday = day before expiry)
    near_exp = opts[opts['DTE'] <= 1].copy()
    if near_exp.empty:
        return pd.DataFrame()

    r = FALLBACK_RATE
    rows = []
    for _, row in near_exp.iterrows():
        dte = int(row['DTE'])
        T = max(dte, 0.5) / 252.0   # 0DTE → half-day; 1DTE → 1 full day
        opt_type = row['Tipo'].lower()
        strike = float(row['strike'])
        close  = float(row['close'])
        if close > 0 and strike > 0:
            iv = _iv(close, spot, strike, T, r, opt_type)
        else:
            iv = 0.30
        rows.append({
            'Ticker': row['ticker'],
            'Tipo':   row['Tipo'],
            'Strike': strike,
            'Ultimo': close,
            'IV':     iv,
            'Delta':  _bs_delta(spot, strike, T, r, iv, opt_type),
            'Gamma':  _bs_gamma(spot, strike, T, r, iv),
            'Tit.':   float(row['quantity']),
            'VolFin': float(row['volume']),
            'DTE':    dte,
        })

    return pd.DataFrame(rows)


def recommend_0dte_strategy(chain_0dte, spot, call_wall, put_wall, gamma_flip, r=0.10):
    """
    Analyse 0DTE options for one asset and recommend the lowest-risk
    defined-risk strategy based on the current GEX regime.

    Parameters
    ----------
    chain_0dte : DataFrame   Rows with DTE==0 (Tipo, Strike, Ultimo, IV, Gamma, Delta, Tit.)
    spot       : float       Current underlying price
    call_wall  : float       GEX call wall (resistance)
    put_wall   : float       GEX put wall (support)
    gamma_flip : float       GEX gamma-flip level
    r          : float       Risk-free rate (annual)

    Returns
    -------
    dict with keys:
        strategy, strikes, max_risk, max_reward, breakevens,
        regime, reason, iv_avg, n_strikes
    or None if insufficient data.
    """
    if chain_0dte.empty or spot <= 0:
        return None

    calls = chain_0dte[chain_0dte['Tipo'] == 'CALL'].sort_values('Strike')


def _is_market_open_now():
    """Return True when current local time is inside configured trade window on a weekday."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H:%M")
    return GEX_TRADE_WINDOW_START <= hm <= GEX_TRADE_WINDOW_END


def _nearest_option_row(df, opt_type, target_strike, target_expiration):
    """Pick the nearest row by strike for option type on target expiration date."""
    if df is None or df.empty:
        return None
    side = df[df['Tipo'].str.upper().str.contains(opt_type.upper())].copy()
    if side.empty:
        return None
    side['Expiration'] = pd.to_datetime(side['Expiration'], errors='coerce')
    side = side.dropna(subset=['Expiration', 'Strike'])
    if side.empty:
        return None
    side = side[side['Expiration'].dt.normalize() == pd.to_datetime(target_expiration).normalize()]
    if side.empty:
        return None
    side['strike_dist'] = (side['Strike'] - float(target_strike)).abs()
    side = side.sort_values(['strike_dist', 'Strike'])
    return side.iloc[0]


def _build_spread_candidates(asset, spot, analysis_result):
    """Build simple Calendar PUT and Iron Condor candidates from live chain snapshot."""
    from b3_options_loader import load_b3_options_data
    from bs_greeks import implied_vol, bs_gamma, bs_delta
    from di1_rate_curve import get_rate_for_date
    from rtd_oi_reader import read_rtd_option_snapshot

    df = load_b3_options_data(asset, spot)
    if df.empty:
        return None

    df = df.copy()
    df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
    df = df.dropna(subset=['Expiration', 'Strike', 'Ultimo'])
    if df.empty:
        return None

    # Prefer live RTD fields (ULT/PEX/CAB) for option price/strike/OI when available.
    tickers = [str(t).strip().upper() for t in df['Ticker'].dropna().astype(str).unique().tolist()]
    rtd = read_rtd_option_snapshot(tickers=tickers, attributes=['ULT', 'PEX', 'CAB'], wait_seconds=1.0, refresh_rounds=3)
    rtd_hits = 0
    if rtd is not None and not rtd.empty:
        rtd['ticker'] = rtd['ticker'].astype(str).str.strip().str.upper()
        rtd_map = rtd.set_index('ticker').to_dict(orient='index')

        def _val(dct, key):
            if dct is None:
                return np.nan
            try:
                return float(dct.get(key, np.nan))
            except (ValueError, TypeError):
                return np.nan

        for i, row in df.iterrows():
            tk = str(row['Ticker']).strip().upper()
            snap = rtd_map.get(tk)
            if snap is None:
                continue
            ult = _val(snap, 'ult')
            pex = _val(snap, 'pex')
            cab = _val(snap, 'cab')
            if np.isfinite(ult) and ult > 0:
                df.at[i, 'Ultimo'] = ult
                rtd_hits += 1
            if np.isfinite(pex) and pex > 0:
                df.at[i, 'Strike'] = pex
            if np.isfinite(cab) and cab > 0:
                df.at[i, 'Tit.'] = cab

        # Recompute IV and Greeks from updated RTD price/strike values.
        new_iv = []
        new_delta = []
        new_gamma = []
        now_date = datetime.now().date()
        for _, row in df.iterrows():
            exp = pd.to_datetime(row['Expiration'], errors='coerce')
            if pd.isna(exp):
                T = 1 / 252.0
                r = 0.10
            else:
                dte = max(int(np.busday_count(now_date, exp.date())), 0)
                T = max(dte / 252.0, 1 / 252.0)
                r = get_rate_for_date(exp)
            opt_type = str(row['Tipo']).lower()
            strike = float(row['Strike']) if np.isfinite(row['Strike']) else np.nan
            price = float(row['Ultimo']) if np.isfinite(row['Ultimo']) else np.nan
            if np.isfinite(price) and price > 0 and np.isfinite(strike) and strike > 0:
                iv = implied_vol(price, spot, strike, T, r, opt_type)
            else:
                iv = float(row['IV']) if 'IV' in row and np.isfinite(row['IV']) else 0.30
            new_iv.append(iv)
            try:
                new_delta.append(bs_delta(spot, strike, T, r, iv, opt_type))
                new_gamma.append(bs_gamma(spot, strike, T, r, iv))
            except Exception:
                new_delta.append(float(row['Delta']) if 'Delta' in row and np.isfinite(row['Delta']) else np.nan)
                new_gamma.append(float(row['Gamma']) if 'Gamma' in row and np.isfinite(row['Gamma']) else np.nan)

        df['IV'] = np.array(new_iv)
        df['Delta'] = np.array(new_delta)
        df['Gamma'] = np.array(new_gamma)

    if rtd_hits > 0:
        print(f"[MARKET CHECK] RTD snapshot applied to {rtd_hits} option rows (ULT/PEX/CAB).")
    else:
        print("[MARKET CHECK] RTD snapshot unavailable for option rows; using B3-derived values.")

    today = pd.Timestamp.now().normalize()
    expirations = sorted([d for d in df['Expiration'].dt.normalize().unique() if d >= today])
    if len(expirations) < 1:
        return None

    near_exp = expirations[0]
    far_exp = expirations[1] if len(expirations) > 1 else expirations[0]

    strike_list = sorted(df['Strike'].dropna().unique())
    if len(strike_list) < 4:
        return None
    diffs = np.diff(strike_list)
    diffs = diffs[diffs > 0]
    strike_step = float(np.median(diffs)) if len(diffs) else 1.0

    put_wall = float(analysis_result.get('put_wall', np.nan))
    call_wall = float(analysis_result.get('call_wall', np.nan))
    if not np.isfinite(put_wall) or not np.isfinite(call_wall):
        return None

    # Calendar PUT (long calendar): sell near-term put, buy farther-term put at same strike.
    cal_short = _nearest_option_row(df, 'PUT', put_wall, near_exp)
    cal_long = _nearest_option_row(df, 'PUT', put_wall, far_exp)
    calendar_put = None
    if cal_short is not None and cal_long is not None:
        short_price = float(cal_short['Ultimo'])
        long_price = float(cal_long['Ultimo'])
        net_debit = max(long_price - short_price, 0.0)
        calendar_put = {
            'short_strike': float(cal_short['Strike']),
            'long_strike': float(cal_long['Strike']),
            'near_exp': pd.Timestamp(near_exp).strftime('%Y-%m-%d'),
            'far_exp': pd.Timestamp(far_exp).strftime('%Y-%m-%d'),
            'near_iv': float(cal_short['IV']) if 'IV' in cal_short else np.nan,
            'far_iv': float(cal_long['IV']) if 'IV' in cal_long else np.nan,
            'short_price': short_price,
            'long_price': long_price,
            'net_debit': net_debit,
        }

    # Iron Condor centered around walls.
    sp = _nearest_option_row(df, 'PUT', put_wall, far_exp)
    lp = _nearest_option_row(df, 'PUT', float(put_wall - strike_step), far_exp)
    sc = _nearest_option_row(df, 'CALL', call_wall, far_exp)
    lc = _nearest_option_row(df, 'CALL', float(call_wall + strike_step), far_exp)
    iron_condor = None
    if sp is not None and lp is not None and sc is not None and lc is not None:
        sp_k = float(sp['Strike'])
        lp_k = float(lp['Strike'])
        sc_k = float(sc['Strike'])
        lc_k = float(lc['Strike'])
        sp_p = float(sp['Ultimo'])
        lp_p = float(lp['Ultimo'])
        sc_p = float(sc['Ultimo'])
        lc_p = float(lc['Ultimo'])
        net_credit = max((sp_p + sc_p) - (lp_p + lc_p), 0.0)
        width_put = max(sp_k - lp_k, 0.0)
        width_call = max(lc_k - sc_k, 0.0)
        max_loss = max(max(width_put, width_call) - net_credit, 0.0)
        be_low = sp_k - net_credit
        be_high = sc_k + net_credit
        iron_condor = {
            'expiry': pd.Timestamp(far_exp).strftime('%Y-%m-%d'),
            'short_put': sp_k,
            'long_put': lp_k,
            'short_call': sc_k,
            'long_call': lc_k,
            'net_credit': net_credit,
            'max_loss': max_loss,
            'breakeven_low': be_low,
            'breakeven_high': be_high,
        }

    # ---------- Optimize parameters for wider profitable coverage ----------
    def _std_norm_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _gex_align(x, target, step_ref):
        if not np.isfinite(x) or not np.isfinite(target):
            return 1.0
        denom = max(step_ref * 2.5, 1e-6)
        return math.exp(-abs(x - target) / denom)

    def _row_price(row, ttm, rate):
        p = float(row['Ultimo']) if np.isfinite(row.get('Ultimo', np.nan)) else np.nan
        if np.isfinite(p) and p > 0:
            return p
        iv = float(row['IV']) if np.isfinite(row.get('IV', np.nan)) and row.get('IV', np.nan) > 0 else 0.30
        k = float(row['Strike'])
        opt_type = str(row['Tipo']).lower()
        return bs_price(spot, k, max(ttm, 1 / 252.0), rate, iv, opt_type)

    near_puts = df[
        df['Tipo'].str.upper().str.contains('PUT')
        & (df['Expiration'].dt.normalize() == pd.to_datetime(near_exp).normalize())
    ].copy()
    far_puts = df[
        df['Tipo'].str.upper().str.contains('PUT')
        & (df['Expiration'].dt.normalize() == pd.to_datetime(far_exp).normalize())
    ].copy()
    far_calls = df[
        df['Tipo'].str.upper().str.contains('CALL')
        & (df['Expiration'].dt.normalize() == pd.to_datetime(far_exp).normalize())
    ].copy()

    near_puts = near_puts.dropna(subset=['Strike']).sort_values('Strike')
    far_puts = far_puts.dropna(subset=['Strike']).sort_values('Strike')
    far_calls = far_calls.dropna(subset=['Strike']).sort_values('Strike')

    now_date = datetime.now().date()
    far_dt = pd.to_datetime(far_exp)
    near_dt = pd.to_datetime(near_exp)
    far_dte = max(int(np.busday_count(now_date, far_dt.date())), 1)
    near_dte = max(int(np.busday_count(now_date, near_dt.date())), 0)
    t_far = max(far_dte / 252.0, 1 / 252.0)
    t_near = max(near_dte / 252.0, 1 / 252.0)
    t_near_to_far = max((far_dte - near_dte) / 252.0, 1 / 252.0)
    r_far = get_rate_for_date(far_dt)
    r_near = get_rate_for_date(near_dt)

    iv_far_ref = float(far_puts['IV'].replace([np.inf, -np.inf], np.nan).dropna().mean()) if not far_puts.empty else 0.25
    if not np.isfinite(iv_far_ref) or iv_far_ref <= 0:
        iv_far_ref = 0.25
    exp_move = max(spot * iv_far_ref * np.sqrt(t_far), strike_step)
    wall_band = max(2.0 * strike_step, 0.8 * exp_move)
    spot_band = max(3.0 * strike_step, 1.5 * exp_move)
    cal_max_wall_distance = max(2.0 * strike_step, 0.45 * exp_move)

    best_calendar_wide = None
    if not near_puts.empty and not far_puts.empty:
        strike_candidates = sorted(set(near_puts['Strike'].unique()).intersection(set(far_puts['Strike'].unique())))
        if not strike_candidates:
            strike_candidates = sorted(far_puts['Strike'].unique())

        # Keep calendar strikes in a practical neighborhood around put wall and spot.
        constrained_candidates = [
            k for k in strike_candidates
            if (abs(float(k) - put_wall) <= wall_band) and (abs(float(k) - spot) <= spot_band)
        ]
        if constrained_candidates:
            strike_candidates = constrained_candidates
        else:
            # Hard fallback: keep only nearest strikes to put wall (do not reopen full range).
            strike_candidates = sorted(
                strike_candidates,
                key=lambda k: abs(float(k) - put_wall)
            )[:max(3, min(7, len(strike_candidates)))]

        x_cal = np.linspace(max(spot - 2.0 * exp_move, strike_list[0] * 0.9),
                            min(spot + 2.0 * exp_move, strike_list[-1] * 1.1), 300)

        best_score = -np.inf
        for k in strike_candidates:
            if abs(float(k) - put_wall) > cal_max_wall_distance:
                continue
            short_row = _nearest_option_row(df, 'PUT', float(k), near_exp)
            long_row = _nearest_option_row(df, 'PUT', float(k), far_exp)
            if short_row is None or long_row is None:
                continue

            short_px = _row_price(short_row, t_near, r_near)
            long_px = _row_price(long_row, t_far, r_far)
            debit = long_px - short_px
            if not np.isfinite(debit) or debit <= 0:
                continue

            iv_long = float(long_row['IV']) if np.isfinite(long_row.get('IV', np.nan)) and long_row.get('IV', np.nan) > 0 else iv_far_ref
            pnl_near = np.array([
                bs_price(s, float(long_row['Strike']), t_near_to_far, r_far, iv_long, 'put')
                - max(float(short_row['Strike']) - s, 0.0)
                - debit
                for s in x_cal
            ])

            positive = pnl_near > 0
            if not np.any(positive):
                continue
            frac_profitable = float(np.mean(positive))
            idx_pos = np.where(positive)[0]
            band = float(x_cal[idx_pos[-1]] - x_cal[idx_pos[0]]) if len(idx_pos) > 1 else 0.0
            peak = float(np.nanmax(pnl_near))
            peak_to_debit = peak / debit if debit > 0 else 0.0
            align = _gex_align(float(k), put_wall, strike_step)
            score = band * (0.4 + frac_profitable) * max(peak_to_debit, 0.1) * align

            if score > best_score:
                best_score = score
                best_calendar_wide = {
                    'short_strike': float(short_row['Strike']),
                    'long_strike': float(long_row['Strike']),
                    'near_exp': pd.Timestamp(near_exp).strftime('%Y-%m-%d'),
                    'far_exp': pd.Timestamp(far_exp).strftime('%Y-%m-%d'),
                    'short_price': float(short_px),
                    'long_price': float(long_px),
                    'net_debit': float(debit),
                    'profit_band_width': band,
                    'profitable_fraction': frac_profitable,
                    'peak_pnl_near_exp': peak,
                    'score': float(score),
                }

        if best_calendar_wide is None and calendar_put is not None:
            # If strict optimization finds no feasible candidate, keep baseline wall-anchored calendar.
            best_calendar_wide = {
                **calendar_put,
                'profit_band_width': np.nan,
                'profitable_fraction': np.nan,
                'peak_pnl_near_exp': np.nan,
                'score': np.nan,
            }

    best_iron_condor_wide = None
    if not far_puts.empty and not far_calls.empty:
        puts_otm = far_puts[far_puts['Strike'] < spot].sort_values('Strike')
        calls_otm = far_calls[far_calls['Strike'] > spot].sort_values('Strike')

        if not puts_otm.empty and not calls_otm.empty:
            put_shorts = puts_otm.tail(min(6, len(puts_otm)))
            call_shorts = calls_otm.head(min(6, len(calls_otm)))

            # Keep short legs near their respective GEX walls while still allowing width optimization.
            put_shorts_c = put_shorts[
                (put_shorts['Strike'] >= (put_wall - wall_band))
                & (put_shorts['Strike'] <= min(spot, put_wall + wall_band))
            ]
            call_shorts_c = call_shorts[
                (call_shorts['Strike'] <= (call_wall + wall_band))
                & (call_shorts['Strike'] >= max(spot, call_wall - wall_band))
            ]
            if not put_shorts_c.empty:
                put_shorts = put_shorts_c
            if not call_shorts_c.empty:
                call_shorts = call_shorts_c

            best_score = -np.inf
            for _, sp_row in put_shorts.iterrows():
                sp_k = float(sp_row['Strike'])
                lp_pool = puts_otm[puts_otm['Strike'] < sp_k].sort_values('Strike', ascending=False).head(3)
                if lp_pool.empty:
                    continue

                for _, sc_row in call_shorts.iterrows():
                    sc_k = float(sc_row['Strike'])
                    lc_pool = calls_otm[calls_otm['Strike'] > sc_k].sort_values('Strike', ascending=True).head(3)
                    if lc_pool.empty:
                        continue

                    for _, lp_row in lp_pool.iterrows():
                        lp_k = float(lp_row['Strike'])
                        for _, lc_row in lc_pool.iterrows():
                            lc_k = float(lc_row['Strike'])
                            sp_px = _row_price(sp_row, t_far, r_far)
                            lp_px = _row_price(lp_row, t_far, r_far)
                            sc_px = _row_price(sc_row, t_far, r_far)
                            lc_px = _row_price(lc_row, t_far, r_far)
                            credit = (sp_px + sc_px) - (lp_px + lc_px)
                            if not np.isfinite(credit) or credit <= 0:
                                continue

                            width_put = sp_k - lp_k
                            width_call = lc_k - sc_k
                            if width_put <= 0 or width_call <= 0:
                                continue

                            max_loss = max(width_put, width_call) - credit
                            if max_loss <= 0:
                                continue

                            be_low = sp_k - credit
                            be_high = sc_k + credit
                            coverage = be_high - be_low
                            rr = credit / max_loss

                            sigma = max(exp_move, strike_step)
                            z_hi = (be_high - spot) / sigma
                            z_lo = (be_low - spot) / sigma
                            pop_proxy = max(_std_norm_cdf(z_hi) - _std_norm_cdf(z_lo), 0.0)

                            gex_fit = _gex_align(sp_k, put_wall, strike_step) * _gex_align(sc_k, call_wall, strike_step)
                            score = coverage * (0.35 + pop_proxy) * min(rr, 3.0) * gex_fit

                            if score > best_score:
                                best_score = score
                                best_iron_condor_wide = {
                                    'expiry': pd.Timestamp(far_exp).strftime('%Y-%m-%d'),
                                    'short_put': sp_k,
                                    'long_put': lp_k,
                                    'short_call': sc_k,
                                    'long_call': lc_k,
                                    'net_credit': float(credit),
                                    'max_loss': float(max_loss),
                                    'breakeven_low': float(be_low),
                                    'breakeven_high': float(be_high),
                                    'profit_width': float(coverage),
                                    'pop_proxy': float(pop_proxy),
                                    'rr': float(rr),
                                    'score': float(score),
                                }

    return {
        'calendar_put': calendar_put,
        'iron_condor': iron_condor,
        'calendar_put_best_wide': best_calendar_wide,
        'iron_condor_best_wide': best_iron_condor_wide,
        'near_exp': pd.Timestamp(near_exp).strftime('%Y-%m-%d'),
        'far_exp': pd.Timestamp(far_exp).strftime('%Y-%m-%d'),
    }


def check_market_open_spread_analysis(asset='BOVA11', force_market_hours=False):
    """
    Test-script helper: run live GEX/IV analysis only when market is open,
    then print candidate Calendar PUT and Iron Condor structures.
    """
    if (not force_market_hours) and (not _is_market_open_now()):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[MARKET CHECK] {now} outside configured trade window {GEX_TRADE_WINDOW_START}-{GEX_TRADE_WINDOW_END}.")
        print("[MARKET CHECK] Skipping live spread check (run this method during market hours).")
        return None

    import MetaTrader5 as mt5
    from mt5_connector import MT5Connector
    from di1_rate_curve import build_di1_curve
    from main import analyze_options

    mt5_conn = MT5Connector()
    build_di1_curve(mt5_conn)

    mt5.symbol_select(asset, True)
    tick = mt5.symbol_info_tick(asset)
    info = mt5.symbol_info(asset)
    spot = 0.0
    if tick is not None and tick.bid > 0 and tick.ask > 0:
        spot = (tick.bid + tick.ask) / 2.0
    elif info is not None and info.last > 0:
        spot = float(info.last)

    if spot <= 0:
        print(f"[MARKET CHECK] Could not fetch live spot for {asset}.")
        return None

    print("\n" + "=" * 78)
    print(f"LIVE MARKET-OPEN SPREAD CHECK -- {asset}")
    print("=" * 78)
    print(f"Spot: {spot:.2f}")

    result = asyncio.run(analyze_options(spot, asset, win_mapper=None, win_symbol='', mt5_conn=mt5_conn))
    if not result:
        print("[MARKET CHECK] analyze_options returned no data.")
        return None

    spreads = _build_spread_candidates(asset, spot, result)
    if not spreads:
        print("[MARKET CHECK] Could not build spread candidates from current chain.")
        return result

    cal = spreads.get('calendar_put')
    ic = spreads.get('iron_condor')
    cal_best = spreads.get('calendar_put_best_wide')
    ic_best = spreads.get('iron_condor_best_wide')

    print("\n[SPREAD CANDIDATES]")
    print(f"Regime: {result.get('regime', 'N/A')}")
    print(f"Call Wall: {result.get('call_wall', np.nan):.2f} | Put Wall: {result.get('put_wall', np.nan):.2f} | Flip: {result.get('gamma_flip', np.nan):.2f}")

    if cal is not None:
        rr_hint = (cal['short_price'] / cal['net_debit']) if cal['net_debit'] > 0 else np.nan
        print("\nCalendar PUT (test candidate)")
        print(f"  Sell PUT {cal['short_strike']:.2f} @ {cal['near_exp']} | Buy PUT {cal['long_strike']:.2f} @ {cal['far_exp']}")
        print(f"  Near IV: {cal['near_iv']*100:.2f}% | Far IV: {cal['far_iv']*100:.2f}%")
        print(f"  Net debit: {cal['net_debit']:.4f} | R/R hint: {rr_hint:.2f}" if np.isfinite(rr_hint) else f"  Net debit: {cal['net_debit']:.4f}")
    else:
        print("\nCalendar PUT (test candidate): unavailable")

    if ic is not None:
        rr = (ic['net_credit'] / ic['max_loss']) if ic['max_loss'] > 0 else np.nan
        print("\nIron Condor (test candidate)")
        print(f"  Sell PUT {ic['short_put']:.2f} / Buy PUT {ic['long_put']:.2f}")
        print(f"  Sell CALL {ic['short_call']:.2f} / Buy CALL {ic['long_call']:.2f}")
        print(f"  Expiry: {ic['expiry']} | Net credit: {ic['net_credit']:.4f} | Max loss: {ic['max_loss']:.4f}")
        print(f"  Break-evens: {ic['breakeven_low']:.2f} / {ic['breakeven_high']:.2f}")
        print(f"  R/R: {rr:.2f}" if np.isfinite(rr) else "  R/R: N/A (max loss near zero)")
    else:
        print("\nIron Condor (test candidate): unavailable")

    print("\n[BEST PARAMETERS - WIDER PROFIT COVERAGE]")
    if cal_best is not None:
        print("Calendar PUT (optimized)")
        print(f"  Sell PUT {cal_best['short_strike']:.2f} @ {cal_best['near_exp']} | Buy PUT {cal_best['long_strike']:.2f} @ {cal_best['far_exp']}")
        print(f"  Net debit: {cal_best['net_debit']:.4f} | Profit-band width: {cal_best['profit_band_width']:.2f}")
        print(f"  Profitable fraction (near-exp MTM): {cal_best['profitable_fraction']*100:.1f}% | Peak near-exp P&L: {cal_best['peak_pnl_near_exp']:.4f}")
    else:
        print("Calendar PUT (optimized): unavailable")

    if ic_best is not None:
        print("Iron Condor (optimized)")
        print(f"  Sell PUT {ic_best['short_put']:.2f} / Buy PUT {ic_best['long_put']:.2f}")
        print(f"  Sell CALL {ic_best['short_call']:.2f} / Buy CALL {ic_best['long_call']:.2f}")
        print(f"  Net credit: {ic_best['net_credit']:.4f} | Max loss: {ic_best['max_loss']:.4f} | R/R: {ic_best['rr']:.2f}")
        print(f"  Profit width: {ic_best['profit_width']:.2f} | POP proxy: {ic_best['pop_proxy']*100:.1f}%")
        print(f"  Break-evens: {ic_best['breakeven_low']:.2f} / {ic_best['breakeven_high']:.2f}")
    else:
        print("Iron Condor (optimized): unavailable")

    print("=" * 78 + "\n")
    return {
        'analysis': result,
        'spreads': spreads,
        'spot': spot,
    }


def plot_spread_payoff_test(asset='BOVA11', force_market_hours=False, output_path=None):
    """
    Test helper: run spread analysis and plot payoff curves for:
            1) Calendar PUT (mark-to-market curves: now, near expiry, far expiry)
      2) Iron Condor (expiry payoff)
    """
    import matplotlib.pyplot as plt
    from di1_rate_curve import get_rate_for_date

    payload = check_market_open_spread_analysis(
        asset=asset,
        force_market_hours=force_market_hours,
    )
    if not payload:
        return None

    spreads = payload.get('spreads', {})
    spot = float(payload.get('spot', np.nan))
    cal = spreads.get('calendar_put') if spreads else None
    ic = spreads.get('iron_condor') if spreads else None

    if cal is None and ic is None:
        print("[PLOT] No spread candidates available to plot.")
        return payload

    strike_anchors = []
    if cal is not None:
        strike_anchors.extend([float(cal['short_strike']), float(cal['long_strike'])])
    if ic is not None:
        strike_anchors.extend([
            float(ic['long_put']), float(ic['short_put']),
            float(ic['short_call']), float(ic['long_call'])
        ])
    if np.isfinite(spot):
        strike_anchors.append(spot)

    lo = min(strike_anchors) * 0.90
    hi = max(strike_anchors) * 1.10
    x = np.linspace(lo, hi, 400)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Calendar PUT: mark-to-market curves using B-S across evaluation horizons.
    ax0 = axes[0]
    if cal is not None:
        def _safe_iv(v, fallback=0.30):
            try:
                x_iv = float(v)
                if np.isfinite(x_iv) and x_iv > 0:
                    return x_iv
            except (ValueError, TypeError):
                pass
            return float(fallback)

        near_dt = pd.to_datetime(cal['near_exp'], errors='coerce')
        far_dt = pd.to_datetime(cal['far_exp'], errors='coerce')
        today = pd.Timestamp.now().normalize()

        near_dte = max(int(np.busday_count(today.date(), near_dt.date())), 0) if pd.notna(near_dt) else 1
        far_dte = max(int(np.busday_count(today.date(), far_dt.date())), 0) if pd.notna(far_dt) else max(near_dte + 1, 1)

        t_now_near = max(near_dte / 252.0, 1 / 252.0)
        t_now_far = max(far_dte / 252.0, 1 / 252.0)
        t_near_far = max((far_dte - near_dte) / 252.0, 1 / 252.0)

        r_near = get_rate_for_date(near_dt) if pd.notna(near_dt) else 0.10
        r_far = get_rate_for_date(far_dt) if pd.notna(far_dt) else 0.10

        iv_near = _safe_iv(cal.get('near_iv', np.nan), fallback=_safe_iv(cal.get('far_iv', np.nan), 0.30))
        iv_far = _safe_iv(cal.get('far_iv', np.nan), fallback=iv_near)

        k_short = float(cal['short_strike'])
        k_long = float(cal['long_strike'])
        debit = float(cal['net_debit'])

        val_now = np.array([
            bs_price(s, k_long, t_now_far, r_far, iv_far, 'put')
            - bs_price(s, k_short, t_now_near, r_near, iv_near, 'put')
            for s in x
        ])
        pnl_now = val_now - debit

        val_near = np.array([
            bs_price(s, k_long, t_near_far, r_far, iv_far, 'put')
            - max(k_short - s, 0.0)
            for s in x
        ])
        pnl_near = val_near - debit

        val_far = np.array([
            max(k_long - s, 0.0) - max(k_short - s, 0.0)
            for s in x
        ])
        pnl_far = val_far - debit

        ax0.plot(x, pnl_now, color='#2563EB', lw=2.1, label='Now (BS MTM)')
        ax0.plot(x, pnl_near, color='#0EA5A4', lw=2.0, ls='--', label='At near expiry (MTM)')
        ax0.plot(x, pnl_far, color='#111827', lw=1.9, ls=':', label='At far expiry')
        ax0.axvline(k_short, color='#DC2626', ls='--', lw=1.2, label='Short PUT strike')
        ax0.axvline(k_long, color='#10B981', ls='--', lw=1.2, label='Long PUT strike')
        ax0.set_title('Calendar PUT (multi-date MTM)')
    else:
        ax0.text(0.5, 0.5, 'Calendar PUT unavailable', ha='center', va='center', transform=ax0.transAxes)
        ax0.set_title('Calendar PUT')
    if np.isfinite(spot):
        ax0.axvline(spot, color='black', ls=':', lw=1.1, label='Spot')
    ax0.axhline(0.0, color='gray', lw=1.0)
    ax0.set_xlabel(f'{asset} Price at Evaluation')
    ax0.set_ylabel('P&L (price units)')
    ax0.grid(alpha=0.25)
    ax0.legend(loc='best', fontsize=8)

    # Iron Condor: exact payoff at expiry from intrinsic legs + net credit.
    ax1 = axes[1]
    if ic is not None:
        lp = float(ic['long_put'])
        sp = float(ic['short_put'])
        sc = float(ic['short_call'])
        lc = float(ic['long_call'])
        credit = float(ic['net_credit'])
        intrinsic_loss = (
            np.maximum(sp - x, 0.0) - np.maximum(lp - x, 0.0)
            + np.maximum(x - sc, 0.0) - np.maximum(x - lc, 0.0)
        )
        pnl_ic = credit - intrinsic_loss
        ax1.plot(x, pnl_ic, color='#7C3AED', lw=2.2, label='Iron Condor P&L')
        ax1.axvline(lp, color='#10B981', ls='--', lw=1.1, label='Long PUT')
        ax1.axvline(sp, color='#DC2626', ls='--', lw=1.1, label='Short PUT')
        ax1.axvline(sc, color='#DC2626', ls='-.', lw=1.1, label='Short CALL')
        ax1.axvline(lc, color='#10B981', ls='-.', lw=1.1, label='Long CALL')
        if np.isfinite(ic.get('breakeven_low', np.nan)):
            ax1.axvline(float(ic['breakeven_low']), color='gray', ls=':', lw=1.0, label='BE low')
        if np.isfinite(ic.get('breakeven_high', np.nan)):
            ax1.axvline(float(ic['breakeven_high']), color='gray', ls=':', lw=1.0, label='BE high')
        ax1.set_title('Iron Condor (expiry payoff)')
    else:
        ax1.text(0.5, 0.5, 'Iron Condor unavailable', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Iron Condor')
    if np.isfinite(spot):
        ax1.axvline(spot, color='black', ls=':', lw=1.1, label='Spot')
    ax1.axhline(0.0, color='gray', lw=1.0)
    ax1.set_xlabel(f'{asset} Price at Expiry')
    ax1.set_ylabel('P&L (price units)')
    ax1.grid(alpha=0.25)
    ax1.legend(loc='best', fontsize=8)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = output_path or f"spread_payoff_test_{asset}_{ts}.png"
    fig.suptitle(f"{asset} - Test Payoff Curves (Calendar PUT + Iron Condor)", fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[PLOT] Saved payoff plot to: {out}")
    payload['plot_file'] = out
    return payload
    puts  = chain_0dte[chain_0dte['Tipo'] == 'PUT'].sort_values('Strike')

    if calls.empty or puts.empty:
        return None

    # Use very small T for 0DTE (half trading day ~ 3.5 hours)
    T = 0.5 / 252.0

    # Available strikes with meaningful OI/volume
    min_oi = chain_0dte['Tit.'].quantile(0.25) if len(chain_0dte) > 4 else 0
    liquid = chain_0dte[chain_0dte['Tit.'] >= max(min_oi, 1)]
    if liquid.empty:
        liquid = chain_0dte

    strikes = sorted(liquid['Strike'].unique())
    n_strikes = len(strikes)
    if n_strikes < 4:
        return None

    iv_avg = float(liquid['IV'].mean())

    # Determine GEX regime
    if np.isfinite(gamma_flip) and spot > 0:
        ratio = spot / gamma_flip
        if ratio > 1.02:
            regime = 'POSITIVE_GAMMA'
        elif ratio < 0.98:
            regime = 'NEGATIVE_GAMMA'
        else:
            regime = 'TRANSITION'
    else:
        regime = 'UNKNOWN'

    # Find ATM strike (closest to spot)
    atm_strike = min(strikes, key=lambda k: abs(k - spot))

    # Find strikes for spreads — pick the most liquid around key levels
    otm_calls = [k for k in strikes if k > spot]
    otm_puts  = [k for k in strikes if k < spot]

    if len(otm_calls) < 2 or len(otm_puts) < 2:
        return None

    # Helper to price a strike using mid-market or B-S
    def get_price(strike, opt_type):
        rows = liquid[(liquid['Strike'] == strike) & (liquid['Tipo'] == opt_type)]
        if not rows.empty and rows.iloc[0]['Ultimo'] > 0:
            return float(rows.iloc[0]['Ultimo'])
        iv_use = float(rows.iloc[0]['IV']) if not rows.empty and rows.iloc[0]['IV'] > 0 else iv_avg
        return bs_price(spot, strike, T, r, iv_use, opt_type.lower())

    # ── Strategy selection based on regime ──────────────────────────

    # Helper: pick the strike N steps OTM from spot
    def nth_otm_call(n):
        return otm_calls[min(n, len(otm_calls) - 1)]
    def nth_otm_put(n):
        return otm_puts[max(len(otm_puts) - 1 - n, 0)]

    if regime == 'POSITIVE_GAMMA':
        # Range-bound: price pinned between walls → Iron Condor
        # Short legs ~2 strikes OTM, wings 1 strike further out
        short_put  = nth_otm_put(1)   # 2nd closest OTM put
        long_put   = nth_otm_put(2)   # 3rd closest
        short_call = nth_otm_call(1)  # 2nd closest OTM call
        long_call  = nth_otm_call(2)  # 3rd closest

        # Try to place short legs near GEX walls for higher probability
        if np.isfinite(put_wall):
            pw_strikes = [k for k in otm_puts if k <= put_wall * 1.01]
            if len(pw_strikes) >= 2:
                short_put = pw_strikes[-1]
                long_put  = pw_strikes[-2]
        if np.isfinite(call_wall):
            cw_strikes = [k for k in otm_calls if k >= call_wall * 0.99]
            if len(cw_strikes) >= 2:
                short_call = cw_strikes[0]
                long_call  = cw_strikes[1]

        # Ensure all 4 strikes are distinct
        if long_put >= short_put:
            long_put = otm_puts[0]
        if long_call <= short_call:
            long_call = otm_calls[-1]

        p_sp = get_price(short_put, 'PUT')
        p_lp = get_price(long_put, 'PUT')
        p_sc = get_price(short_call, 'CALL')
        p_lc = get_price(long_call, 'CALL')

        credit   = (p_sp - p_lp) + (p_sc - p_lc)
        put_width  = abs(short_put - long_put)
        call_width = abs(long_call - short_call)
        max_width  = max(put_width, call_width)
        max_risk   = max_width - credit
        max_reward = credit
        be_low  = short_put - credit
        be_high = short_call + credit

        return {
            'strategy': 'Iron Condor',
            'legs': f"Sell {short_put:.2f}P / Buy {long_put:.2f}P + Sell {short_call:.2f}C / Buy {long_call:.2f}C",
            'strikes': (long_put, short_put, short_call, long_call),
            'max_risk': max(max_risk, 0.01),
            'max_reward': max(max_reward, 0),
            'breakevens': (be_low, be_high),
            'regime': regime,
            'reason': 'Positive gamma = range-bound; sell premium between GEX walls',
            'iv_avg': iv_avg,
            'n_strikes': n_strikes,
        }

    elif regime == 'NEGATIVE_GAMMA':
        # Trending / volatile: use a Debit Spread in the trend direction
        # Near put wall = expect bounce → Bull Call Spread
        # Near call wall = expect rejection → Bear Put Spread
        dist_to_pw = abs(spot - put_wall) if np.isfinite(put_wall) else 999
        dist_to_cw = abs(spot - call_wall) if np.isfinite(call_wall) else 999

        if dist_to_pw <= dist_to_cw:
            # Closer to put wall → bullish bounce → Bull Call Spread
            buy_k  = nth_otm_call(0)   # 1st OTM call (slightly OTM)
            sell_k = nth_otm_call(2)   # 3rd OTM call (wider wing)
            if buy_k == sell_k:
                sell_k = nth_otm_call(1)
            p_buy  = get_price(buy_k, 'CALL')
            p_sell = get_price(sell_k, 'CALL')
            debit  = p_buy - p_sell
            width  = sell_k - buy_k
            max_risk   = max(debit, 0.01)
            max_reward = width - debit
            be = buy_k + debit
            strat_name = 'Bull Call Spread'
            legs_str   = f"Buy {buy_k:.2f}C / Sell {sell_k:.2f}C"
            strikes_t  = (buy_k, sell_k)
            reason     = 'Negative gamma near put wall: expect bounce, limited-risk bullish'
            bes        = (be,)
        else:
            # Closer to call wall → bearish rejection → Bear Put Spread
            buy_k  = nth_otm_put(0)   # 1st OTM put (slightly OTM)
            sell_k = nth_otm_put(2)   # 3rd OTM put (wider wing)
            if buy_k == sell_k:
                sell_k = nth_otm_put(1)
            p_buy  = get_price(buy_k, 'PUT')
            p_sell = get_price(sell_k, 'PUT')
            debit  = p_buy - p_sell
            width  = buy_k - sell_k
            max_risk   = max(debit, 0.01)
            max_reward = width - debit
            be = buy_k - debit
            strat_name = 'Bear Put Spread'
            legs_str   = f"Buy {buy_k:.2f}P / Sell {sell_k:.2f}P"
            strikes_t  = (sell_k, buy_k)
            reason     = 'Negative gamma near call wall: expect rejection, limited-risk bearish'
            bes        = (be,)

        return {
            'strategy': strat_name,
            'legs': legs_str,
            'strikes': strikes_t,
            'max_risk': max_risk,
            'max_reward': max(max_reward, 0),
            'breakevens': bes,
            'regime': regime,
            'reason': reason,
            'iv_avg': iv_avg,
            'n_strikes': n_strikes,
        }

    else:
        # TRANSITION / UNKNOWN → Iron Butterfly (cheapest defined-risk, profits from pin)
        # Sell ATM straddle + buy nearby OTM wings (~2-3 strikes out)
        wing_put  = nth_otm_put(2)    # 3rd OTM put
        wing_call = nth_otm_call(2)   # 3rd OTM call

        p_sc = get_price(atm_strike, 'CALL')
        p_sp = get_price(atm_strike, 'PUT')
        p_wc = get_price(wing_call, 'CALL')
        p_wp = get_price(wing_put, 'PUT')

        credit   = (p_sc + p_sp) - (p_wc + p_wp)
        put_wing_w  = atm_strike - wing_put
        call_wing_w = wing_call - atm_strike
        max_width   = max(put_wing_w, call_wing_w)
        max_risk    = max_width - credit
        max_reward  = credit
        be_low  = atm_strike - credit
        be_high = atm_strike + credit

        return {
            'strategy': 'Iron Butterfly',
            'legs': f"Sell {atm_strike:.2f}C+P / Buy {wing_put:.2f}P + {wing_call:.2f}C",
            'strikes': (wing_put, atm_strike, wing_call),
            'max_risk': max(max_risk, 0.01),
            'max_reward': max(max_reward, 0),
            'breakevens': (be_low, be_high),
            'regime': regime,
            'reason': 'Transition/unknown regime: sell ATM premium with defined wings',
            'iv_avg': iv_avg,
            'n_strikes': n_strikes,
        }


def _compute_gex_walls_for_day(date_str, spot):
    """
    Load B3 options for a specific date, compute GEX columns,
    then compute weekly walls + combined gamma flip.
    Returns (call_wall, put_wall, gamma_flip, support_zones, resist_zones) in BOVA11 terms, or Nones.
    """
    from b3_options_loader import load_b3_options_data

    df = load_b3_options_data("BOVA11", spot, date=date_str)
    if df.empty:
        return None, None, None, pd.DataFrame(), pd.DataFrame()

    # Compute customer GEX per row (same formula as main.py)
    sign = np.where(df['Tipo'].str.upper().str.contains('PUT'), -1.0, 1.0)
    df['GEX_customer'] = df['Gamma'] * (spot ** 2) * df['Tit.'] * sign

    weekly = compute_weekly_walls(df, spot)
    if not weekly:
        return None, None, None, pd.DataFrame(), pd.DataFrame()

    # Combined walls across weeks (same as main.py)
    all_gex = pd.concat([w['gex_by_strike'] for w in weekly if not w['gex_by_strike'].empty])
    if all_gex.empty:
        return None, None, None, pd.DataFrame(), pd.DataFrame()

    combined = all_gex.groupby('Strike', as_index=False)['GEX_customer'].sum()
    above = combined[combined['Strike'] >= spot]
    below = combined[combined['Strike'] <= spot]

    call_wall = float(above.loc[above['GEX_customer'].idxmax(), 'Strike']) if not above.empty else np.nan
    put_wall = float(below.loc[below['GEX_customer'].abs().idxmax(), 'Strike']) if not below.empty else np.nan

    gamma_flip = find_gamma_flip(df, spot)

    # Compute support / resistance zones for entry levels
    resist_zones, support_zones = _select_significant_zones(combined, spot, top_n=3, zone_pct=0.04)

    return call_wall, put_wall, gamma_flip, support_zones, resist_zones


def simulate_day(day_label, win_bars, bova_bars, call_wall_bova, put_wall_bova,
                 gamma_flip_bova, mapper, support_zones=None, resist_zones=None):
    """
    Simulate one trading day of the DCA GEX strategy using intraday bars.

    Parameters
    ----------
    day_label    : str   e.g. "2026-04-07"
    win_bars     : DataFrame with columns: time, open, high, low, close
    bova_bars    : DataFrame (parallel) — used to derive BOVA spot
    call_wall_bova, put_wall_bova, gamma_flip_bova : GEX levels in BOVA terms
    mapper       : KalmanPriceMapper for BOVA↔WIN conversion

    Returns dict with trade results for the day.
    """
    vol_initial = max(int(MARGIN_BUDGET / (MARGIN_PER_LOT * FIB_TOTAL)), int(GEX_ORDER_VOLUME))
    risk_r = MARGIN_BUDGET * GEX_SL_RISK_PCT
    dca_step_r = MARGIN_BUDGET * GEX_DCA_LOSS_STEP_PCT
    activation_r = MARGIN_BUDGET * GEX_TRAILING_ACTIVATION_PCT

    # Convert GEX levels to WIN
    win_call_wall = mapper.bova11_to_ind(call_wall_bova) if np.isfinite(call_wall_bova) else np.nan
    win_put_wall  = mapper.bova11_to_ind(put_wall_bova) if np.isfinite(put_wall_bova) else np.nan

    # Entry levels from nearest support / resistance zones
    if support_zones is not None and not support_zones.empty:
        below = support_zones[support_zones['Strike'] <= mapper.ind_to_bova11((win_bars['high'].iloc[0] + win_bars['low'].iloc[0]) / 2.0)]
        entry_buy_bova = float(below['Strike'].max()) if not below.empty else np.nan
    else:
        entry_buy_bova = put_wall_bova * (1.0 + GEX_WALL_PROXIMITY_PCT) if np.isfinite(put_wall_bova) else np.nan
    if resist_zones is not None and not resist_zones.empty:
        above = resist_zones[resist_zones['Strike'] >= mapper.ind_to_bova11((win_bars['high'].iloc[0] + win_bars['low'].iloc[0]) / 2.0)]
        entry_sell_bova = float(above['Strike'].min()) if not above.empty else np.nan
    else:
        entry_sell_bova = call_wall_bova * (1.0 - GEX_WALL_PROXIMITY_PCT) if np.isfinite(call_wall_bova) else np.nan

    trades = []  # list of completed trade dicts
    daily_realized_pnl = 0.0  # track daily loss for circuit breaker
    daily_loss_limit = MARGIN_BUDGET * GEX_MAX_DAILY_LOSS_PCT

    # ── Signal diagnostics (per-day) ─────────────────────────
    sig_counter = {'BUY': 0, 'SELL': 0, 'BREAKOUT_UP': 0, 'BREAKOUT_DOWN': 0,
                   'NEUTRAL': 0, 'TRANSITION': 0, 'OTHER': 0}
    diag = {'eligible_buy_bars': 0, 'eligible_sell_bars': 0,
            'rejected_no_target': 0, 'rejected_window': 0}

    # TP levels in WIN terms — nearest opposite-side S/R zone, fallback to wall
    # BUY enters at nearest support → exits at nearest resistance above entry
    tp_buy_bova = np.nan
    if resist_zones is not None and not resist_zones.empty and np.isfinite(entry_buy_bova):
        above = resist_zones[resist_zones['Strike'] > entry_buy_bova]
        if not above.empty:
            tp_buy_bova = float(above['Strike'].min())
    if not np.isfinite(tp_buy_bova):
        tp_buy_bova = call_wall_bova
    win_tp_buy = mapper.bova11_to_ind(tp_buy_bova) if np.isfinite(tp_buy_bova) else np.nan

    # SELL enters at nearest resistance → exits at nearest support below entry
    tp_sell_bova = np.nan
    if support_zones is not None and not support_zones.empty and np.isfinite(entry_sell_bova):
        below = support_zones[support_zones['Strike'] < entry_sell_bova]
        if not below.empty:
            tp_sell_bova = float(below['Strike'].max())
    if not np.isfinite(tp_sell_bova):
        tp_sell_bova = put_wall_bova
    win_tp_sell = mapper.bova11_to_ind(tp_sell_bova) if np.isfinite(tp_sell_bova) else np.nan

    # Per-side state
    for side_label, is_buy, entry_bova in [
        ('BUY', True, entry_buy_bova),
        ('SELL', False, entry_sell_bova)
    ]:
        # Note: entry_bova is no longer required — zones are computed per-bar.
        # Daily loss circuit breaker
        if daily_realized_pnl <= -daily_loss_limit:
            continue

        executed = False
        trail = None

        buy_confirm_ticks = 0
        sell_confirm_ticks = 0
        _confirm_threshold = max(1, int((GEX_CONFIRMATION_MINUTES * 60) / 300))  # assume 5-min bars

        for i, bar in win_bars.iterrows():
            win_mid = (bar['high'] + bar['low']) / 2.0
            bova_spot = mapper.ind_to_bova11(win_mid)

            signal = generate_gex_trade_signals(
                bova_spot, gamma_flip_bova, call_wall_bova, put_wall_bova,
                support_zones=support_zones, resist_zones=resist_zones)
            sig = signal['signal']
            strength = signal['strength']

            # diagnostics
            sig_counter[sig if sig in sig_counter else 'OTHER'] = sig_counter.get(sig if sig in sig_counter else 'OTHER', 0) + 1
            if is_buy and sig == 'BUY' and strength >= GEX_MIN_SIGNAL_STRENGTH:
                diag['eligible_buy_bars'] += 1
            if (not is_buy) and sig == 'SELL' and strength >= GEX_MIN_SIGNAL_STRENGTH:
                diag['eligible_sell_bars'] += 1

            buy_candidate = (sig == 'BUY' and strength >= GEX_MIN_SIGNAL_STRENGTH)
            sell_candidate = (sig == 'SELL' and strength >= GEX_MIN_SIGNAL_STRENGTH)

            buy_confirm_ticks = buy_confirm_ticks + 1 if buy_candidate else 0
            sell_confirm_ticks = sell_confirm_ticks + 1 if sell_candidate else 0

            buy_confirm_ok = (not GEX_REQUIRE_5M_CONFIRMATION) or (buy_confirm_ticks >= _confirm_threshold)
            sell_confirm_ok = (not GEX_REQUIRE_5M_CONFIRMATION) or (sell_confirm_ticks >= _confirm_threshold)
            neutral_ok = (not GEX_NEUTRAL_ONLY) or _is_neutral_setup(
                bova_spot, gamma_flip_bova, call_wall_bova, put_wall_bova, GEX_NEUTRAL_MAX_FLIP_DISTANCE_PCT
            )

            # ── Check entry ──────────────────────────────────
            if not executed:
                # Time window filter
                bar_time_hm = bar['time'].strftime("%H:%M") if hasattr(bar['time'], 'strftime') else "00:00"
                if not (GEX_TRADE_WINDOW_START <= bar_time_hm <= GEX_TRADE_WINDOW_END):
                    continue

                trigger = False
                # Dynamic per-bar zone selection: nearest support BELOW spot,
                # nearest resistance ABOVE spot. Fire when bar pierces zone
                # OR comes within GEX_WALL_PROXIMITY_PCT of it.
                bar_low_bova = mapper.ind_to_bova11(bar['low'])
                bar_high_bova = mapper.ind_to_bova11(bar['high'])
                prox = GEX_WALL_PROXIMITY_PCT

                nearest_sup_bova = np.nan
                if support_zones is not None and not support_zones.empty:
                    sup_below = support_zones[support_zones['Strike'] <= bar_high_bova * (1.0 + prox)]
                    if not sup_below.empty:
                        nearest_sup_bova = float(sup_below['Strike'].max())
                if not np.isfinite(nearest_sup_bova) and np.isfinite(put_wall_bova):
                    nearest_sup_bova = put_wall_bova

                nearest_res_bova = np.nan
                if resist_zones is not None and not resist_zones.empty:
                    res_above = resist_zones[resist_zones['Strike'] >= bar_low_bova * (1.0 - prox)]
                    if not res_above.empty:
                        nearest_res_bova = float(res_above['Strike'].min())
                if not np.isfinite(nearest_res_bova) and np.isfinite(call_wall_bova):
                    nearest_res_bova = call_wall_bova

                buy_level_win = mapper.bova11_to_ind(nearest_sup_bova) if np.isfinite(nearest_sup_bova) else np.nan
                sell_level_win = mapper.bova11_to_ind(nearest_res_bova) if np.isfinite(nearest_res_bova) else np.nan

                # Touch within proximity band
                buy_touch = (np.isfinite(buy_level_win)
                             and bar['low'] <= buy_level_win * (1.0 + prox))
                sell_touch = (np.isfinite(sell_level_win)
                              and bar['high'] >= sell_level_win * (1.0 - prox))

                if (is_buy and sig == 'BUY' and strength >= GEX_MIN_SIGNAL_STRENGTH
                        and buy_confirm_ok and neutral_ok and buy_touch):
                    trigger = True
                    # Enter at zone level if reached, else at bar low (best fill)
                    entry_price = align_tick(max(buy_level_win, bar['low']))
                elif (not is_buy and sig == 'SELL' and strength >= GEX_MIN_SIGNAL_STRENGTH
                        and sell_confirm_ok and neutral_ok and sell_touch):
                    trigger = True
                    entry_price = align_tick(min(sell_level_win, bar['high']))

                # Per-trade TP: opposite side zone (must be on profit side)
                if trigger:
                    if is_buy and np.isfinite(sell_level_win) and sell_level_win > entry_price:
                        trade_tp_win = sell_level_win
                    elif (not is_buy) and np.isfinite(buy_level_win) and buy_level_win < entry_price:
                        trade_tp_win = buy_level_win
                    else:
                        # No valid opposite zone → fall back to wall TP
                        trade_tp_win = win_tp_buy if is_buy else win_tp_sell
                    # Reject if TP is on the wrong side
                    if is_buy and (not np.isfinite(trade_tp_win) or trade_tp_win <= entry_price):
                        trigger = False
                    elif (not is_buy) and (not np.isfinite(trade_tp_win) or trade_tp_win >= entry_price):
                        trigger = False

                if trigger:
                    vol = vol_initial
                    sl_dist = (risk_r / vol) / PNL_PER_POINT if vol > 0 else 0
                    sl_dist = round(sl_dist / TICK_SIZE) * TICK_SIZE
                    sl_dist = max(sl_dist, GEX_MIN_SL_POINTS)  # enforce minimum SL floor
                    if is_buy:
                        sl_price = align_tick(entry_price - sl_dist)
                    else:
                        sl_price = align_tick(entry_price + sl_dist)

                    # TP from the per-bar dynamic zone (set above when trigger
                    # fired). Already validated to be on the profit side.
                    tp_price = np.nan
                    if GEX_TP_AT_OPPOSITE_WALL and np.isfinite(trade_tp_win):
                        tp_price = align_tick(trade_tp_win)

                    trail = {
                        'entry': entry_price, 'avg_entry': entry_price,
                        'vol': vol, 'total_vol': vol,
                        'best': entry_price, 'active': False,
                        'sl_dist': sl_dist, 'sl_price': sl_price,
                        'tp_price': tp_price,
                        'dca_count': 0, 'entry_time': bar['time'],
                        'positions': [{'entry': entry_price, 'vol': vol, 'label': 'Initial'}],
                    }
                    executed = True
                continue

            # ── Position active: check SL, DCA, trailing ─────
            if is_buy:
                check_price = bar['low']  # worst for long
                best_price = bar['high']  # best for long
            else:
                check_price = bar['high']  # worst for short
                best_price = bar['low']   # best for short

            avg = trail['avg_entry']
            tvol = trail['total_vol']
            pnl_per_pt = tvol * PNL_PER_POINT

            # Update best
            if is_buy:
                trail['best'] = max(trail['best'], best_price)
            else:
                trail['best'] = min(trail['best'], best_price)

            # ── SL check ─────────────────────────────────────
            stopped = False
            if is_buy and check_price <= trail['sl_price']:
                stopped = True
                exit_price = trail['sl_price']
            elif not is_buy and check_price >= trail['sl_price']:
                stopped = True
                exit_price = trail['sl_price']

            if stopped:
                if is_buy:
                    pnl_pts = exit_price - avg
                else:
                    pnl_pts = avg - exit_price
                pnl_r = pnl_pts * tvol * PNL_PER_POINT
                daily_realized_pnl += pnl_r

                trades.append({
                    'day': day_label, 'side': side_label,
                    'entry': trail['entry'], 'avg_entry': avg,
                    'exit': exit_price, 'exit_time': bar['time'],
                    'entry_time': trail['entry_time'],
                    'vol': tvol, 'dca': trail['dca_count'],
                    'pnl_pts': pnl_pts, 'pnl_r': pnl_r,
                    'exit_type': 'STOP',
                    'trailing': trail['active'],
                    'call_wall': call_wall_bova, 'put_wall': put_wall_bova,
                    'gamma_flip': gamma_flip_bova,
                })
                break

            # ── TP check (opposite wall) ─────────────────────
            tp_hit = False
            tp = trail.get('tp_price', np.nan)
            if np.isfinite(tp):
                if is_buy and best_price >= tp:
                    tp_hit = True
                    exit_price = align_tick(tp)
                elif not is_buy and best_price <= tp:
                    tp_hit = True
                    exit_price = align_tick(tp)

            if tp_hit:
                if is_buy:
                    pnl_pts = exit_price - avg
                else:
                    pnl_pts = avg - exit_price
                pnl_r = pnl_pts * tvol * PNL_PER_POINT
                daily_realized_pnl += pnl_r

                trades.append({
                    'day': day_label, 'side': side_label,
                    'entry': trail['entry'], 'avg_entry': avg,
                    'exit': exit_price, 'exit_time': bar['time'],
                    'entry_time': trail['entry_time'],
                    'vol': tvol, 'dca': trail['dca_count'],
                    'pnl_pts': pnl_pts, 'pnl_r': pnl_r,
                    'exit_type': 'TP',
                    'trailing': trail['active'],
                    'call_wall': call_wall_bova, 'put_wall': put_wall_bova,
                    'gamma_flip': gamma_flip_bova,
                })
                break

            # ── DCA (before trailing activates) ──────────────
            if not trail['active'] and trail['dca_count'] < GEX_DCA_MAX_ORDERS:
                if is_buy:
                    loss_pts = avg - check_price
                else:
                    loss_pts = check_price - avg
                loss_r = loss_pts * tvol * PNL_PER_POINT

                if loss_r > 0:
                    levels_crossed = int(loss_r / dca_step_r) if dca_step_r > 0 else 0
                    needed = levels_crossed - trail['dca_count']
                    while needed > 0:
                        if trail['dca_count'] >= GEX_DCA_MAX_ORDERS:
                            break
                        fib_idx = min(trail['dca_count'], len(FIB_SEQ) - 1)
                        dca_vol = trail['vol'] * FIB_SEQ[fib_idx]

                        # Margin cap check
                        margin_used = tvol * MARGIN_PER_LOT
                        margin_needed = dca_vol * MARGIN_PER_LOT
                        if margin_used + margin_needed > MARGIN_BUDGET:
                            break

                        dca_price = align_tick(check_price)
                        new_total = tvol + dca_vol
                        new_avg = (avg * tvol + dca_price * dca_vol) / new_total

                        # Recalculate SL to bound aggregate risk
                        pnl_per_pt_new = new_total * PNL_PER_POINT
                        new_sl_dist = (risk_r / pnl_per_pt_new) if pnl_per_pt_new > 0 else 0
                        new_sl_dist = round(new_sl_dist / TICK_SIZE) * TICK_SIZE
                        new_sl_dist = max(new_sl_dist, GEX_MIN_SL_POINTS)  # enforce minimum SL floor

                        # Risk guard: skip DCA if min SL floor causes actual risk > 2× budget
                        actual_risk = new_sl_dist * pnl_per_pt_new
                        if actual_risk > risk_r * 2:
                            break

                        if is_buy:
                            new_sl = align_tick(new_avg - new_sl_dist)
                        else:
                            new_sl = align_tick(new_avg + new_sl_dist)

                        trail['avg_entry'] = new_avg
                        trail['total_vol'] = new_total
                        trail['sl_dist'] = new_sl_dist
                        trail['sl_price'] = new_sl
                        trail['dca_count'] += 1
                        trail['positions'].append({
                            'entry': dca_price, 'vol': dca_vol,
                            'label': f"DCA#{trail['dca_count']}"
                        })
                        avg = new_avg
                        tvol = new_total

                        # Recalculate loss & needed after avg entry changed
                        if is_buy:
                            loss_pts = avg - check_price
                        else:
                            loss_pts = check_price - avg
                        loss_r = loss_pts * tvol * PNL_PER_POINT
                        levels_crossed = int(loss_r / dca_step_r) if dca_step_r > 0 else 0
                        needed = levels_crossed - trail['dca_count']

            # ── Trailing activation ──────────────────────────
            if is_buy:
                profit_pts = trail['best'] - avg
            else:
                profit_pts = avg - trail['best']
            profit_r = profit_pts * tvol * PNL_PER_POINT

            if not trail['active'] and profit_r >= activation_r:
                trail['active'] = True

            if trail['active']:
                # Use tighter trailing distance (factor of original SL)
                trail_dist = trail['sl_dist'] * GEX_TRAILING_DISTANCE_FACTOR
                trail_dist = max(trail_dist, GEX_MIN_SL_POINTS)  # never below floor
                if is_buy:
                    new_sl = align_tick(trail['best'] - trail_dist)
                    if new_sl > trail['sl_price']:
                        trail['sl_price'] = new_sl
                else:
                    new_sl = align_tick(trail['best'] + trail_dist)
                    if new_sl < trail['sl_price']:
                        trail['sl_price'] = new_sl

        else:
            # Day ended without stop — mark to market at close
            if executed and trail is not None:
                close_price = align_tick(win_bars.iloc[-1]['close'])
                if is_buy:
                    pnl_pts = close_price - trail['avg_entry']
                else:
                    pnl_pts = trail['avg_entry'] - close_price
                pnl_r = pnl_pts * trail['total_vol'] * PNL_PER_POINT

                trades.append({
                    'day': day_label, 'side': side_label,
                    'entry': trail['entry'], 'avg_entry': trail['avg_entry'],
                    'exit': close_price, 'exit_time': win_bars.iloc[-1]['time'],
                    'entry_time': trail['entry_time'],
                    'vol': trail['total_vol'], 'dca': trail['dca_count'],
                    'pnl_pts': pnl_pts, 'pnl_r': pnl_r,
                    'exit_type': 'EOD',
                    'trailing': trail['active'],
                    'call_wall': call_wall_bova, 'put_wall': put_wall_bova,
                    'gamma_flip': gamma_flip_bova,
                })

    # Print per-day diagnostic
    print(f"  [diag] sig: BUY={sig_counter['BUY']} SELL={sig_counter['SELL']} "
          f"BO↑={sig_counter['BREAKOUT_UP']} BO↓={sig_counter['BREAKOUT_DOWN']} "
          f"NEU={sig_counter['NEUTRAL']} TR={sig_counter['TRANSITION']} | "
          f"eligible bars: BUY={diag['eligible_buy_bars']} SELL={diag['eligible_sell_bars']}")
    print(f"  [diag] entry/TP (BOVA): BUY {entry_buy_bova:.2f}->{tp_buy_bova:.2f}  "
          f"SELL {entry_sell_bova:.2f}->{tp_sell_bova:.2f}")

    return trades


def main(lookback_days: int = 5):
    import MetaTrader5 as mt5
    from mt5_connector import MT5Connector
    from di1_rate_curve import build_di1_curve
    from kalman_price_mapper import build_ind_bova11_mapper_intraday

    mt5_conn = MT5Connector()
    build_di1_curve(mt5_conn)

    # Resolve current WIN symbol + expiring contract for data fallback
    try:
        (_, win_symbol), expiring_sym = mt5_conn.get_symbol_futures("*WIN*", include_expiring=True)
        print(f"[i] WIN contract: {win_symbol}")
        if expiring_sym:
            print(f"[i] Expiring contract (data fallback): {expiring_sym}")
    except Exception as e:
        print(f"[!] Could not resolve WIN futures contract: {e}")
        return

    # Build Kalman mapper on 15-min intraday data (more bars = reliable fit)
    # Trading symbol first; expiring contract as historical-data fallback
    mapper = None
    syms_to_try = [win_symbol]
    if expiring_sym and expiring_sym != win_symbol:
        syms_to_try.append(expiring_sym)
    for sym in syms_to_try:
        try:
            mapper = build_ind_bova11_mapper_intraday(
                mt5_conn, ind_symbol=sym, bova11_symbol="BOVA11", max_days=10)
            print(f"[i] Kalman mapper built (intraday 15m) using {sym}")
            break
        except Exception as e:
            print(f"[!] Mapper with {sym} failed: {e}")
    if mapper is None:
        print("[!] Could not build Kalman mapper — aborting")
        return

    # ── Determine backtest dates: last week (trading days with cached data) ──
    # Cached files in .b3_cache: COTAHIST_D{DDMMYYYY}.ZIP
    cache_dir = os.path.join(SCRIPT_DIR, ".b3_cache")
    cached_dates = []
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith("COTAHIST_D") and f.endswith(".ZIP"):
            try:
                ds = f.replace("COTAHIST_D", "").replace(".ZIP", "")
                d = datetime.strptime(ds, "%d%m%Y")
                cached_dates.append(d)
            except ValueError:
                continue

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # Last N business days (Mon-Fri) before today that have cached data
    last_week = [d for d in cached_dates
                 if d < today and d.weekday() < 5]
    last_week = sorted(last_week)[-int(lookback_days):]  # last N trading days

    if not last_week:
        print(f"[!] No cached B3 data for the last {lookback_days} trading days. Run main.py first to cache data.")
        return

    print(f"\n{'='*80}")
    print(f"  DCA GEX STRATEGY — HISTORICAL BACKTEST")
    print(f"  Dates: {last_week[0].strftime('%Y-%m-%d')} to {last_week[-1].strftime('%Y-%m-%d')}")
    print(f"  WIN Symbol: {win_symbol}")
    print(f"{'='*80}")
    print(f"  Parameters:")
    print(f"    Margin Budget     = R$ {MARGIN_BUDGET:,.2f} ({GEX_MARGIN_FREE_PCT:.0%} of R${FREE_MARGIN:,.0f})")
    print(f"    SL Risk           = R$ {MARGIN_BUDGET * GEX_SL_RISK_PCT:,.2f} ({GEX_SL_RISK_PCT:.0%})")
    print(f"    Min SL Floor      = {GEX_MIN_SL_POINTS} pts")
    print(f"    DCA Step          = R$ {MARGIN_BUDGET * GEX_DCA_LOSS_STEP_PCT:,.2f} ({GEX_DCA_LOSS_STEP_PCT:.0%})")
    print(f"    DCA Max Orders    = {GEX_DCA_MAX_ORDERS}")
    print(f"    Trailing Activate = R$ {MARGIN_BUDGET * GEX_TRAILING_ACTIVATION_PCT:,.2f} ({GEX_TRAILING_ACTIVATION_PCT:.0%})")
    print(f"    Trailing Factor   = {GEX_TRAILING_DISTANCE_FACTOR:.0%} of SL")
    print(f"    TP at Opp. Wall   = {'Yes' if GEX_TP_AT_OPPOSITE_WALL else 'No'}")
    print(f"    Daily Loss Cap    = {GEX_MAX_DAILY_LOSS_PCT:.0%} of budget")
    print(f"    Trade Window      = {GEX_TRADE_WINDOW_START} - {GEX_TRADE_WINDOW_END}")
    print(f"    Fib Sequence      = {FIB_SEQ[:GEX_DCA_MAX_ORDERS]}")
    print(f"    Wall Proximity    = {GEX_WALL_PROXIMITY_PCT:.1%}")
    print(f"{'='*80}\n")

    all_trades = []
    all_0dte_recs = []   # [{day, asset, spot, rec_dict}, ...]

    for day_dt in last_week:
        day_str = day_dt.strftime("%Y-%m-%d")
        b3_date = day_dt.strftime("%Y-%m-%d")
        print(f"\n{'─'*80}")
        print(f"  DAY: {day_str} ({day_dt.strftime('%A')})")
        print(f"{'─'*80}")

        # Fetch BOVA11 spot for this day (daily bar)
        # Get daily bars up to this day
        bova_daily = mt5_conn.get_data("BOVA11", mt5.TIMEFRAME_D1, 10, 0)
        if bova_daily is None or bova_daily.empty:
            print(f"  [!] No BOVA11 daily data — skipping")
            continue

        # Find the bar matching this day (or nearest prior)
        bova_daily['date'] = bova_daily['time'].dt.date
        day_bar = bova_daily[bova_daily['date'] <= day_dt.date()]
        if day_bar.empty:
            print(f"  [!] No BOVA11 bar for {day_str} — skipping")
            continue
        day_open_bova = float(day_bar.iloc[-1]['open'])
        day_close_bova = float(day_bar.iloc[-1]['close'])
        spot_bova = day_open_bova  # Use open for GEX calc (pre-market)

        print(f"  BOVA11: open={spot_bova:.2f}, close={day_close_bova:.2f}")

        # ── Compute GEX walls from cached B3 data ────────────
        call_wall, put_wall, gamma_flip, sup_zones, res_zones = _compute_gex_walls_for_day(b3_date, spot_bova)
        if call_wall is None or not np.isfinite(call_wall):
            print(f"  [!] Could not compute GEX walls — skipping")
            continue

        win_cw = mapper.bova11_to_ind(call_wall) if np.isfinite(call_wall) else np.nan
        win_pw = mapper.bova11_to_ind(put_wall) if np.isfinite(put_wall) else np.nan
        win_gf = mapper.bova11_to_ind(gamma_flip) if np.isfinite(gamma_flip) else np.nan

        print(f"  GEX Walls (BOVA): CW={call_wall:.2f}  PW={put_wall:.2f}  Flip={gamma_flip:.2f}")
        print(f"  GEX Walls (WIN) : CW={win_cw:.0f}  PW={win_pw:.0f}  Flip={win_gf:.0f}")

        # ── Fetch intraday 5-min WIN bars for this day ───────
        # We fetch enough bars to cover last week then filter to the target day
        bars_per_day = 90  # ~7.5 hours × 12 bars/hour at 5-min
        total_bars = bars_per_day * 10  # extra buffer
        try:
            win_intra, _ = mt5_conn.get_historical_futures_data("*WIN*", mt5.TIMEFRAME_M5, total_bars, 0)
        except Exception:
            win_intra = None

        if win_intra is None or win_intra.empty:
            print(f"  [!] No intraday WIN data — skipping")
            continue

        win_intra['date'] = win_intra['time'].dt.date
        day_bars = win_intra[win_intra['date'] == day_dt.date()].copy()

        if day_bars.empty:
            print(f"  [!] No 5-min bars for {day_str} — skipping")
            continue

        # Also fetch BOVA11 intraday for mapper validation
        bova_intra = mt5_conn.get_data("BOVA11", mt5.TIMEFRAME_M5, total_bars, 0)
        if bova_intra is not None and not bova_intra.empty:
            bova_intra['date'] = bova_intra['time'].dt.date
            bova_day = bova_intra[bova_intra['date'] == day_dt.date()].copy()
        else:
            bova_day = pd.DataFrame()

        print(f"  WIN bars: {len(day_bars)} (5-min) | "
              f"Range: {day_bars.iloc[0]['time'].strftime('%H:%M')}-{day_bars.iloc[-1]['time'].strftime('%H:%M')}")
        print(f"  WIN range: {day_bars['low'].min():.0f} - {day_bars['high'].max():.0f}")

        # ── Run simulation ────────────────────────────────────
        day_trades = simulate_day(
            day_label=day_str,
            win_bars=day_bars.reset_index(drop=True),
            bova_bars=bova_day,
            call_wall_bova=call_wall,
            put_wall_bova=put_wall,
            gamma_flip_bova=gamma_flip,
            mapper=mapper,
            support_zones=sup_zones,
            resist_zones=res_zones,
        )

        if not day_trades:
            print(f"  → No signals triggered (no trades)")
        for t in day_trades:
            pnl_sign = '+' if t['pnl_r'] >= 0 else ''
            print(f"  → {t['side']:4s} @ {t['entry']:>8,.0f} → {t['exit']:>8,.0f} "
                  f"({t['exit_type']:4s}) | vol={t['vol']} dca={t['dca']} | "
                  f"P&L: {pnl_sign}R$ {t['pnl_r']:>8,.2f} "
                  f"({t['pnl_pts']:>+,.0f} pts) "
                  f"| trail={'Y' if t['trailing'] else 'N'}")
            all_trades.append(t)

        # ── 0DTE strategy recommendation per asset ────────────
        print(f"\n  {'0DTE / 1DTE STRATEGY RECOMMENDATIONS':^60}")
        print(f"  {'─'*60}")
        day_has_0dte = False
        for asset in ASSET_SYMBOL:
            try:
                if asset == "BOVA11":
                    a_spot = spot_bova
                    a_cw, a_pw, a_gf = call_wall, put_wall, gamma_flip
                else:
                    a_daily = mt5_conn.get_data(asset, mt5.TIMEFRAME_D1, 10, 0)
                    if a_daily is None or a_daily.empty:
                        print(f"  [{asset}] No daily data — skipped")
                        continue
                    a_daily['date'] = a_daily['time'].dt.date
                    a_day_bar = a_daily[a_daily['date'] <= day_dt.date()]
                    if a_day_bar.empty:
                        print(f"  [{asset}] No bar for {day_str} — skipped")
                        continue
                    a_spot = float(a_day_bar.iloc[-1]['open'])
                    if a_spot <= 0:
                        print(f"  [{asset}] Invalid spot ({a_spot}) — skipped")
                        continue
                    # Compute lightweight GEX walls from the full chain
                    from b3_options_loader import load_b3_options_data as _load_full
                    a_df = _load_full(asset, a_spot, date=b3_date)
                    if a_df.empty:
                        a_cw, a_pw, a_gf = np.nan, np.nan, np.nan
                    else:
                        sign_a = np.where(a_df['Tipo'].str.upper().str.contains('PUT'), -1.0, 1.0)
                        a_df['GEX_customer'] = a_df['Gamma'] * (a_spot ** 2) * a_df['Tit.'] * sign_a
                        comb = a_df.groupby('Strike', as_index=False)['GEX_customer'].sum()
                        abv = comb[comb['Strike'] >= a_spot]
                        blw = comb[comb['Strike'] <= a_spot]
                        a_cw = float(abv.loc[abv['GEX_customer'].idxmax(), 'Strike']) if not abv.empty else np.nan
                        a_pw = float(blw.loc[blw['GEX_customer'].abs().idxmax(), 'Strike']) if not blw.empty else np.nan
                        a_gf = find_gamma_flip(a_df, a_spot)

                chain_0 = _load_0dte_chain(asset, a_spot, b3_date)
                if chain_0.empty:
                    print(f"  [{asset}] No 0DTE chain found for {b3_date}")
                    continue
                rec = recommend_0dte_strategy(chain_0, a_spot, a_cw, a_pw, a_gf)
                if rec:
                    # Determine entry type: 0DTE (expiry day) vs 1DTE (day before)
                    min_dte = int(chain_0['DTE'].min())
                    dte_label = '0DTE' if min_dte == 0 else '1DTE'
                    weekday = day_dt.strftime('%a')
                    all_0dte_recs.append({
                        'day': day_str, 'asset': asset,
                        'spot': a_spot, 'dte_label': dte_label,
                        'weekday': weekday, **rec
                    })
                    day_has_0dte = True
                    rr = rec['max_reward'] / rec['max_risk'] if rec['max_risk'] > 0 else 0
                    bes_str = ' / '.join(f"{b:.2f}" for b in rec['breakevens'])
                    print(f"  [{asset}] {dte_label} | Spot={a_spot:.2f} | Regime: {rec['regime']}")
                    print(f"    Strategy : {rec['strategy']}  |  Legs: {rec['legs']}")
                    print(f"    Reason   : {rec['reason']}")
                    print(f"    MaxRisk  : R$ {rec['max_risk']:.2f}  |  MaxReward: R$ {rec['max_reward']:.2f}  |  R:R 1:{rr:.1f}")
                    print(f"    Breakeven: {bes_str}  |  Avg IV: {rec['iv_avg']*100:.1f}%  |  Strikes: {rec['n_strikes']}")
                else:
                    print(f"  [{asset}] 0DTE chain found but no viable strategy")
            except Exception as e:
                print(f"  [{asset}] 0DTE error: {e}")
        if not day_has_0dte:
            print(f"  No 0DTE/1DTE recommendations for {day_str}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*100}")
    print(f"  BACKTEST SUMMARY — {last_week[0].strftime('%Y-%m-%d')} to {last_week[-1].strftime('%Y-%m-%d')}")
    print(f"{'='*100}")

    if not all_trades:
        print("  No trades executed during the period.")
        return

    df_trades = pd.DataFrame(all_trades)

    total_pnl = df_trades['pnl_r'].sum()
    n_trades = len(df_trades)
    n_wins = (df_trades['pnl_r'] > 0).sum()
    n_losses = (df_trades['pnl_r'] < 0).sum()
    n_be = (df_trades['pnl_r'] == 0).sum()
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
    avg_win = df_trades[df_trades['pnl_r'] > 0]['pnl_r'].mean() if n_wins > 0 else 0
    avg_loss = df_trades[df_trades['pnl_r'] < 0]['pnl_r'].mean() if n_losses > 0 else 0
    max_win = df_trades['pnl_r'].max()
    max_loss = df_trades['pnl_r'].min()
    profit_factor = abs(avg_win * n_wins / (avg_loss * n_losses)) if n_losses > 0 and avg_loss != 0 else float('inf')

    # Per-side breakdown
    buys = df_trades[df_trades['side'] == 'BUY']
    sells = df_trades[df_trades['side'] == 'SELL']

    print(f"\n  {'TRADE LOG':^96}")
    print(f"  {'─'*96}")
    print(f"  {'Day':<12} {'Side':<5} {'Entry':>8} {'Avg':>8} {'Exit':>8} {'Type':>4} {'Vol':>4} "
          f"{'DCA':>3} {'Trail':>5} {'P&L R$':>12} {'P&L pts':>10}")
    print(f"  {'─'*96}")
    for _, t in df_trades.iterrows():
        p = '+' if t['pnl_r'] >= 0 else ''
        print(f"  {t['day']:<12} {t['side']:<5} {t['entry']:>8,.0f} {t['avg_entry']:>8,.0f} "
              f"{t['exit']:>8,.0f} {t['exit_type']:>4} {t['vol']:>4} {t['dca']:>3} "
              f"{'Y' if t['trailing'] else 'N':>5} "
              f"{p}R$ {t['pnl_r']:>9,.2f} {t['pnl_pts']:>+10,.0f}")
    print(f"  {'─'*96}")
    print(f"  {'TOTAL':<60} {'' :>18} R$ {total_pnl:>+10,.2f}")

    print(f"\n  {'STATISTICS':^70}")
    print(f"  {'─'*70}")
    print(f"    Total Trades       : {n_trades}")
    print(f"    Wins / Losses / BE : {n_wins} / {n_losses} / {n_be}")
    print(f"    Win Rate           : {win_rate:.1f}%")
    print(f"    Total P&L          : R$ {total_pnl:+,.2f}")
    print(f"    Avg Win            : R$ {avg_win:+,.2f}")
    print(f"    Avg Loss           : R$ {avg_loss:+,.2f}")
    print(f"    Max Win            : R$ {max_win:+,.2f}")
    print(f"    Max Loss           : R$ {max_loss:+,.2f}")
    print(f"    Profit Factor      : {profit_factor:.2f}")
    print(f"    ROI on Budget      : {total_pnl / MARGIN_BUDGET * 100:+.1f}%")

    if not buys.empty:
        print(f"\n    BUY side  : {len(buys)} trades, P&L R$ {buys['pnl_r'].sum():+,.2f}, "
              f"win rate {(buys['pnl_r'] > 0).mean()*100:.0f}%")
    if not sells.empty:
        print(f"    SELL side : {len(sells)} trades, P&L R$ {sells['pnl_r'].sum():+,.2f}, "
              f"win rate {(sells['pnl_r'] > 0).mean()*100:.0f}%")

    # Per-day P&L
    print(f"\n  {'DAILY P&L':^70}")
    print(f"  {'─'*70}")
    daily = df_trades.groupby('day')['pnl_r'].sum()
    cum = 0
    for day, pnl in daily.items():
        cum += pnl
        bar = '█' * max(1, int(abs(pnl) / 20))
        color = '+' if pnl >= 0 else '-'
        print(f"    {day}  R$ {pnl:>+9,.2f}  cum: R$ {cum:>+9,.2f}  {color}{bar}")
    print(f"  {'─'*70}")
    print(f"    CUMULATIVE: R$ {cum:>+9,.2f}  ({cum / MARGIN_BUDGET * 100:+.1f}% ROI)")

    # ══════════════════════════════════════════════════════════════════
    # 0DTE OPTION STRATEGY RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════
    if all_0dte_recs:
        print(f"\n\n{'='*110}")
        print(f"  SHORT-DATED OPTION STRATEGIES — LOWEST-RISK RECOMMENDATIONS PER ASSET")
        print(f"  (0DTE = expiry day / Friday | 1DTE = day before expiry / Thursday)")
        print(f"{'='*110}")

        # Group by asset for a summary table
        for asset in ASSET_SYMBOL:
            asset_recs = [r for r in all_0dte_recs if r['asset'] == asset]
            if not asset_recs:
                continue

            print(f"\n  {'─'*106}")
            print(f"  {asset:^106}")
            print(f"  {'─'*106}")
            print(f"  {'Day':<12} {'':>3} {'DTE':<5} {'Spot':>8} {'Regime':<16} {'Strategy':<18} "
                  f"{'Legs':<38} {'MaxRisk':>8} {'MaxRwd':>8}")
            print(f"  {'─'*106}")

            for r in asset_recs:
                print(f"  {r['day']:<12} {r.get('weekday',''):>3} {r.get('dte_label','?'):<5} {r['spot']:>8.2f} {r['regime']:<16} "
                      f"{r['strategy']:<18} {r['legs']:<38} "
                      f"R${r['max_risk']:>6.2f} R${r['max_reward']:>6.2f}")

            # Print detail for last day (most recent / actionable)
            last = asset_recs[-1]
            print(f"\n    Latest ({last['day']} {last.get('weekday','')}, {last.get('dte_label','')}):")
            print(f"      Strategy   : {last['strategy']}")
            print(f"      Regime     : {last['regime']}")
            print(f"      Reason     : {last['reason']}")
            print(f"      Legs       : {last['legs']}")
            print(f"      Max Risk   : R$ {last['max_risk']:.2f} per contract")
            print(f"      Max Reward : R$ {last['max_reward']:.2f} per contract")
            bes_str = ' / '.join(f"{b:.2f}" for b in last['breakevens'])
            print(f"      Breakevens : {bes_str}")
            print(f"      Avg IV     : {last['iv_avg']*100:.1f}%")
            print(f"      Liquid Strikes : {last['n_strikes']}")
            rr = last['max_reward'] / last['max_risk'] if last['max_risk'] > 0 else 0
            print(f"      Risk/Reward: 1:{rr:.1f}")

        # Aggregated regime overview
        print(f"\n  {'─'*96}")
        print(f"  {'REGIME SUMMARY':^96}")
        print(f"  {'─'*96}")
        for asset in ASSET_SYMBOL:
            asset_recs = [r for r in all_0dte_recs if r['asset'] == asset]
            if not asset_recs:
                continue
            regimes = [r['regime'] for r in asset_recs]
            most_common = max(set(regimes), key=regimes.count)
            strats = [r['strategy'] for r in asset_recs]
            most_strat = max(set(strats), key=strats.count)
            avg_iv = np.mean([r['iv_avg'] for r in asset_recs])
            print(f"    {asset:<8} Dominant regime: {most_common:<18} "
                  f"Best strategy: {most_strat:<20} Avg IV: {avg_iv*100:.1f}%")

        print(f"  {'─'*96}")
        print()
    else:
        print(f"\n  [i] No 0DTE options data available for the backtest period.\n")


# ═══════════════════════════════════════════════════════════════════════
#  STANDALONE 0DTE STRATEGY RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_0dte_only():
    """Run ONLY the 0DTE strategy recommendations (no DCA backtest)."""
    import MetaTrader5 as mt5
    from mt5_connector import MT5Connector

    mt5_conn = MT5Connector()

    # ── Determine dates from cached B3 data ──────────────────────────
    cache_dir = os.path.join(SCRIPT_DIR, ".b3_cache")
    cached_dates = []
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith("COTAHIST_D") and f.endswith(".ZIP"):
            try:
                ds = f.replace("COTAHIST_D", "").replace(".ZIP", "")
                d = datetime.strptime(ds, "%d%m%Y")
                cached_dates.append(d)
            except ValueError:
                continue

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    last_week = [d for d in cached_dates if d < today and d.weekday() < 5]
    last_week = sorted(last_week)[-5:]

    if not last_week:
        print("[!] No cached B3 data. Run main.py first to cache data.")
        return

    print(f"\n{'='*110}")
    print(f"  0DTE / 1DTE STRATEGY RECOMMENDATIONS ONLY")
    print(f"  Dates: {last_week[0].strftime('%Y-%m-%d')} to {last_week[-1].strftime('%Y-%m-%d')}")
    print(f"  Assets: {', '.join(ASSET_SYMBOL)}")
    print(f"{'='*110}\n")

    all_0dte_recs = []

    for day_dt in last_week:
        day_str = day_dt.strftime("%Y-%m-%d")
        b3_date = day_dt.strftime("%Y-%m-%d")
        print(f"\n{'─'*80}")
        print(f"  DAY: {day_str} ({day_dt.strftime('%A')})")
        print(f"{'─'*80}")

        day_has_0dte = False
        for asset in ASSET_SYMBOL:
            try:
                # Get spot price
                a_daily = mt5_conn.get_data(asset, mt5.TIMEFRAME_D1, 10, 0)
                if a_daily is None or a_daily.empty:
                    print(f"  [{asset}] No daily data — skipped")
                    continue
                a_daily['date'] = a_daily['time'].dt.date
                a_day_bar = a_daily[a_daily['date'] <= day_dt.date()]
                if a_day_bar.empty:
                    print(f"  [{asset}] No bar for {day_str} — skipped")
                    continue
                a_spot = float(a_day_bar.iloc[-1]['open'])
                if a_spot <= 0:
                    print(f"  [{asset}] Invalid spot ({a_spot}) — skipped")
                    continue

                # Compute GEX walls
                if asset == "BOVA11":
                    a_cw, a_pw, a_gf, _, _ = _compute_gex_walls_for_day(b3_date, a_spot)
                else:
                    from b3_options_loader import load_b3_options_data as _load_full
                    a_df = _load_full(asset, a_spot, date=b3_date)
                    if a_df.empty:
                        a_cw, a_pw, a_gf = np.nan, np.nan, np.nan
                    else:
                        sign_a = np.where(a_df['Tipo'].str.upper().str.contains('PUT'), -1.0, 1.0)
                        a_df['GEX_customer'] = a_df['Gamma'] * (a_spot ** 2) * a_df['Tit.'] * sign_a
                        comb = a_df.groupby('Strike', as_index=False)['GEX_customer'].sum()
                        abv = comb[comb['Strike'] >= a_spot]
                        blw = comb[comb['Strike'] <= a_spot]
                        a_cw = float(abv.loc[abv['GEX_customer'].idxmax(), 'Strike']) if not abv.empty else np.nan
                        a_pw = float(blw.loc[blw['GEX_customer'].abs().idxmax(), 'Strike']) if not blw.empty else np.nan
                        a_gf = find_gamma_flip(a_df, a_spot)

                print(f"  [{asset}] Spot={a_spot:.2f}  CW={a_cw if np.isfinite(a_cw) else 'N/A'}  "
                      f"PW={a_pw if np.isfinite(a_pw) else 'N/A'}  Flip={a_gf if np.isfinite(a_gf) else 'N/A'}")

                # Load 0DTE chain and recommend
                chain_0 = _load_0dte_chain(asset, a_spot, b3_date)
                if chain_0.empty:
                    print(f"    → No 0DTE/1DTE chain found")
                    continue

                print(f"    → 0DTE chain: {len(chain_0)} rows, DTE range: {chain_0['DTE'].min()}-{chain_0['DTE'].max()}")
                rec = recommend_0dte_strategy(chain_0, a_spot, a_cw, a_pw, a_gf)
                if rec:
                    min_dte = int(chain_0['DTE'].min())
                    dte_label = '0DTE' if min_dte == 0 else '1DTE'
                    weekday = day_dt.strftime('%a')
                    all_0dte_recs.append({
                        'day': day_str, 'asset': asset,
                        'spot': a_spot, 'dte_label': dte_label,
                        'weekday': weekday, **rec
                    })
                    day_has_0dte = True
                    rr = rec['max_reward'] / rec['max_risk'] if rec['max_risk'] > 0 else 0
                    bes_str = ' / '.join(f"{b:.2f}" for b in rec['breakevens'])
                    print(f"    ✓ {dte_label} | Regime: {rec['regime']}")
                    print(f"      Strategy : {rec['strategy']}  |  Legs: {rec['legs']}")
                    print(f"      Reason   : {rec['reason']}")
                    print(f"      MaxRisk  : R$ {rec['max_risk']:.2f}  |  MaxReward: R$ {rec['max_reward']:.2f}  |  R:R 1:{rr:.1f}")
                    print(f"      Breakeven: {bes_str}  |  Avg IV: {rec['iv_avg']*100:.1f}%  |  Strikes: {rec['n_strikes']}")
                else:
                    print(f"    → Chain found but no viable strategy (need ≥4 liquid strikes)")
            except Exception as e:
                print(f"  [{asset}] 0DTE error: {e}")

        if not day_has_0dte:
            print(f"  → No 0DTE/1DTE recommendations for {day_str}")

    # ── Final summary table ───────────────────────────────────────────
    if all_0dte_recs:
        print(f"\n\n{'='*110}")
        print(f"  SUMMARY — SHORT-DATED OPTION STRATEGIES")
        print(f"{'='*110}")

        for asset in ASSET_SYMBOL:
            asset_recs = [r for r in all_0dte_recs if r['asset'] == asset]
            if not asset_recs:
                continue
            print(f"\n  {'─'*106}")
            print(f"  {asset:^106}")
            print(f"  {'─'*106}")
            print(f"  {'Day':<12} {'':>3} {'DTE':<5} {'Spot':>8} {'Regime':<16} {'Strategy':<18} "
                  f"{'Legs':<38} {'MaxRisk':>8} {'MaxRwd':>8}")
            print(f"  {'─'*106}")
            for r in asset_recs:
                print(f"  {r['day']:<12} {r.get('weekday',''):>3} {r.get('dte_label','?'):<5} {r['spot']:>8.2f} "
                      f"{r['regime']:<16} {r['strategy']:<18} {r['legs']:<38} "
                      f"R${r['max_risk']:>6.2f} R${r['max_reward']:>6.2f}")

        print(f"\n  {'─'*96}")
        print(f"  {'REGIME SUMMARY':^96}")
        print(f"  {'─'*96}")
        for asset in ASSET_SYMBOL:
            asset_recs = [r for r in all_0dte_recs if r['asset'] == asset]
            if not asset_recs:
                continue
            regimes = [r['regime'] for r in asset_recs]
            most_common = max(set(regimes), key=regimes.count)
            strats = [r['strategy'] for r in asset_recs]
            most_strat = max(set(strats), key=strats.count)
            avg_iv = np.mean([r['iv_avg'] for r in asset_recs])
            print(f"    {asset:<8} Dominant regime: {most_common:<18} "
                  f"Best strategy: {most_strat:<20} Avg IV: {avg_iv*100:.1f}%")
        print(f"  {'─'*96}\n")
    else:
        print(f"\n  [!] No 0DTE options data available for any date in the period.\n")


if __name__ == "__main__":
    if "--plot-spread-test" in sys.argv:
        asset = "BOVA11"
        force = "--force-market-open-check" in sys.argv
        out = None
        if "--asset" in sys.argv:
            idx = sys.argv.index("--asset")
            if idx + 1 < len(sys.argv):
                asset = sys.argv[idx + 1].strip().upper()
        if "--out" in sys.argv:
            idx = sys.argv.index("--out")
            if idx + 1 < len(sys.argv):
                out = sys.argv[idx + 1].strip()
        plot_spread_payoff_test(asset=asset, force_market_hours=force, output_path=out)
    elif "--market-open-check" in sys.argv:
        asset = "BOVA11"
        force = "--force-market-open-check" in sys.argv
        if "--asset" in sys.argv:
            idx = sys.argv.index("--asset")
            if idx + 1 < len(sys.argv):
                asset = sys.argv[idx + 1].strip().upper()
        check_market_open_spread_analysis(asset=asset, force_market_hours=force)
    else:
        days = 5
        if "--days" in sys.argv:
            idx = sys.argv.index("--days")
            if idx + 1 < len(sys.argv):
                try:
                    days = int(sys.argv[idx + 1])
                except ValueError:
                    print(f"[!] Invalid --days value: {sys.argv[idx + 1]}; using {days}")
        main(lookback_days=days)

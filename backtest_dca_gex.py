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
    ASSET_SYMBOL,
)
from gex_utils import compute_weekly_walls, find_gamma_flip, generate_gex_trade_signals
from main import _select_significant_zones
from bs_greeks import bs_price

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

    # TP levels in WIN terms (opposite wall)
    win_tp_buy = win_call_wall  # BUY exits at call wall
    win_tp_sell = win_put_wall  # SELL exits at put wall

    # Per-side state
    for side_label, is_buy, entry_bova in [
        ('BUY', True, entry_buy_bova),
        ('SELL', False, entry_sell_bova)
    ]:
        if not np.isfinite(entry_bova):
            continue
        # Daily loss circuit breaker
        if daily_realized_pnl <= -daily_loss_limit:
            continue

        executed = False
        trail = None

        for i, bar in win_bars.iterrows():
            win_mid = (bar['high'] + bar['low']) / 2.0
            bova_spot = mapper.ind_to_bova11(win_mid)

            signal = generate_gex_trade_signals(
                bova_spot, gamma_flip_bova, call_wall_bova, put_wall_bova,
                support_zones=support_zones, resist_zones=resist_zones)
            sig = signal['signal']
            strength = signal['strength']

            # ── Check entry ──────────────────────────────────
            if not executed:
                # Time window filter
                bar_time_hm = bar['time'].strftime("%H:%M") if hasattr(bar['time'], 'strftime') else "00:00"
                if not (GEX_TRADE_WINDOW_START <= bar_time_hm <= GEX_TRADE_WINDOW_END):
                    continue

                trigger = False
                if is_buy and sig == 'BUY' and strength >= GEX_MIN_SIGNAL_STRENGTH:
                    trigger = True
                    entry_price = align_tick(bar['high'])  # conservative: buy at high
                elif not is_buy and sig == 'SELL' and strength >= GEX_MIN_SIGNAL_STRENGTH:
                    trigger = True
                    entry_price = align_tick(bar['low'])  # conservative: sell at low

                if trigger:
                    vol = vol_initial
                    sl_dist = (risk_r / vol) / PNL_PER_POINT if vol > 0 else 0
                    sl_dist = round(sl_dist / TICK_SIZE) * TICK_SIZE
                    sl_dist = max(sl_dist, GEX_MIN_SL_POINTS)  # enforce minimum SL floor
                    if is_buy:
                        sl_price = align_tick(entry_price - sl_dist)
                    else:
                        sl_price = align_tick(entry_price + sl_dist)

                    # TP at opposite wall
                    tp_price = np.nan
                    if GEX_TP_AT_OPPOSITE_WALL:
                        tp_price = win_tp_buy if is_buy else win_tp_sell

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

    return trades


def main():
    import MetaTrader5 as mt5
    from mt5_connector import MT5Connector
    from di1_rate_curve import build_di1_curve
    from kalman_price_mapper import build_ind_bova11_mapper_intraday

    mt5_conn = MT5Connector()
    build_di1_curve(mt5_conn)

    # Resolve current WIN symbol
    try:
        _, win_symbol = mt5_conn.get_symbol_futures("*WIN*")
        print(f"[i] WIN contract: {win_symbol}")
    except Exception as e:
        win_symbol = "WIN$N"
        print(f"[i] Falling back to {win_symbol}: {e}")

    # Build Kalman mapper on 15-min intraday data (more bars = reliable fit)
    mapper = None
    for ind_sym in ["WIN$N", win_symbol]:
        try:
            mapper = build_ind_bova11_mapper_intraday(
                mt5_conn, ind_symbol=ind_sym, bova11_symbol="BOVA11", max_days=10)
            print(f"[i] Kalman mapper built (intraday 15m) using {ind_sym}")
            break
        except Exception as e:
            print(f"[!] Mapper with {ind_sym} failed: {e}")
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
    # Last 5 business days (Mon-Fri) before today that have cached data
    last_week = [d for d in cached_dates
                 if d < today and d.weekday() < 5]
    last_week = sorted(last_week)[-5:]  # last 5 trading days

    if not last_week:
        print("[!] No cached B3 data for last week. Run main.py first to cache data.")
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
        win_intra = mt5_conn.get_data(win_symbol, mt5.TIMEFRAME_M5, total_bars, 0)
        if win_intra is None or win_intra.empty:
            # Try continuous contract
            win_intra = mt5_conn.get_data("WIN$N", mt5.TIMEFRAME_M5, total_bars, 0)

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
        for asset in ASSET_SYMBOL:
            try:
                if asset == "BOVA11":
                    a_spot = spot_bova
                    a_cw, a_pw, a_gf = call_wall, put_wall, gamma_flip
                else:
                    a_daily = mt5_conn.get_data(asset, mt5.TIMEFRAME_D1, 10, 0)
                    if a_daily is None or a_daily.empty:
                        continue
                    a_daily['date'] = a_daily['time'].dt.date
                    a_day_bar = a_daily[a_daily['date'] <= day_dt.date()]
                    if a_day_bar.empty:
                        continue
                    a_spot = float(a_day_bar.iloc[-1]['open'])
                    if a_spot <= 0:
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
            except Exception as e:
                pass  # skip silently — 0DTE is informational only

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


if __name__ == "__main__":
    main()

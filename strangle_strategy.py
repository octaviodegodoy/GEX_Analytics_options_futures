# -*- coding: utf-8 -*-
"""
Strangle Strategy Builder
--------------------------
Constructs strangle setups (long or short) using GEX call/put walls as strikes.

Core idea:
  - SHORT strangle: Sell OTM call (at/near call wall) + Sell OTM put (at/near
    put wall).  Best in POSITIVE GAMMA regime — dealers dampen vol, spot stays
    between walls, theta harvest is reliable.
  - LONG strangle:  Buy OTM call + Buy OTM put.  Best in NEGATIVE GAMMA regime
    — dealers amplify moves, directional breakout expected on one side.
  - TRANSITION:     Long strangle with reduced-size note (flip instability).

Expiry selection:
  Nearest valid weekly expiry from GEX data is used by default.  A second
  expiry can be passed as far-expiry; if absent the near expiry is reused.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes helpers (self-contained, mirrors flyagonal_strategy.py)
# ---------------------------------------------------------------------------

def _bs_price(S, K, T, r, sigma, option_type='call'):
    """European option price via Black-Scholes. T in years, r continuous."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Return delta, gamma, theta, vega for one leg."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100.0  # per 1% IV move

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252.0

    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta),
        'vega': float(vega),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _snap_to_strike(target, df):
    """Snap target price to nearest available strike in the chain."""
    strikes = df['Strike'].unique()
    if len(strikes) == 0:
        return target
    return float(strikes[np.argmin(np.abs(strikes - target))])


def _get_iv(df, strike, expiry_dt, option_type):
    """Retrieve IV from chain for a specific strike + expiry + type."""
    mask = (
        (df['Strike'] == strike) &
        (pd.to_datetime(df['Expiration']).dt.date == expiry_dt.date()) &
        (df['Tipo'].str.upper().str.contains(option_type.upper()))
    )
    matched = df.loc[mask, 'IV']
    if matched.empty:
        return np.nan
    return float(matched.iloc[0])


def _fallback_iv(df, strike, option_type):
    """IV from nearest strike of same type, any expiry."""
    sub = df[df['Tipo'].str.upper().str.contains(option_type.upper())].copy()
    if sub.empty:
        return np.nan
    sub = sub.copy()
    sub['_dist'] = (sub['Strike'] - strike).abs()
    return float(sub.sort_values('_dist').iloc[0]['IV'])


def _pnl_profile(legs, spot, T, r, n_points=60):
    """P&L profile at expiry across ±12% range of spot."""
    lo = spot * 0.88
    hi = spot * 1.12
    prices = np.linspace(lo, hi, n_points)

    entry_cost = sum(
        (-1.0 if l['action'] == 'BUY' else 1.0) * l['qty'] * l['price']
        for l in legs
    )

    pnl_list = []
    for S in prices:
        value = 0.0
        for leg in legs:
            sign = 1.0 if leg['action'] == 'BUY' else -1.0
            if leg['type'] == 'call':
                val = max(S - leg['strike'], 0.0)
            else:
                val = max(leg['strike'] - S, 0.0)
            value += sign * leg['qty'] * val
        pnl_list.append(value + entry_cost)

    return pd.DataFrame({'price': prices, 'pnl': pnl_list})


def _find_break_evens(pnl_df):
    """Find prices where P&L crosses zero."""
    if pnl_df.empty:
        return []
    pnl = pnl_df['pnl'].values
    prices = pnl_df['price'].values
    sign = np.sign(pnl)
    sign[sign == 0] = 1
    crossings = np.where(np.diff(sign) != 0)[0]
    bes = []
    for i in crossings:
        p0, p1 = prices[i], prices[i + 1]
        v0, v1 = pnl[i], pnl[i + 1]
        if v1 != v0:
            be = p0 + (0 - v0) * (p1 - p0) / (v1 - v0)
            bes.append(float(be))
    return bes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_strangle(df, spot, weekly_results, call_wall, put_wall, regime, r=0.10):
    """
    Build a strangle trade from GEX data.

    Parameters
    ----------
    df : pd.DataFrame
        Full options chain (Strike, IV, Gamma, Tipo, Tit., Expiration).
    spot : float
        Current underlying price.
    weekly_results : list[dict]
        Output from compute_weekly_walls().
    call_wall : float
        GEX call wall strike (used as short/long call strike).
    put_wall : float
        GEX put wall strike (used as short/long put strike).
    regime : str
        Current gamma regime string.
    r : float
        Annualised risk-free rate (default 10% for Brazil SELIC-ish).

    Returns
    -------
    dict or None
        Strategy specification with legs, Greeks, P&L profile, and rationale.
        Returns None if insufficient data.
    """
    valid_weeks = [w for w in weekly_results if not w['gex_by_strike'].empty]
    if not valid_weeks:
        return None

    wk = valid_weeks[0]
    dte = max(wk['dte'], 1)
    T = dte / 252.0

    # Direction from regime
    regime_up = regime.upper()
    if 'POSITIVE' in regime_up:
        action = 'SELL'
        direction = 'SHORT'
        suitability = 'IDEAL'
        rationale = (
            'Positive gamma regime — dealers dampen vol, spot expected to '
            'stay between walls. Sell both wings to harvest theta.'
        )
    elif 'NEGATIVE' in regime_up:
        action = 'BUY'
        direction = 'LONG'
        suitability = 'IDEAL'
        rationale = (
            'Negative gamma regime — dealers amplify moves, directional '
            'breakout expected. Buy both wings for convex payoff.'
        )
    else:  # TRANSITION / UNKNOWN
        action = 'BUY'
        direction = 'LONG'
        suitability = 'SUBOPTIMAL'
        rationale = (
            'Transition zone — gamma flip unstable. Long strangle provides '
            'optionality in both directions; reduce size.'
        )

    # Strike selection: snap GEX walls to actual chain strikes
    call_strike_raw = call_wall if np.isfinite(call_wall) else spot * 1.02
    put_strike_raw = put_wall if np.isfinite(put_wall) else spot * 0.98

    call_strike = _snap_to_strike(call_strike_raw, df[df['Tipo'].str.upper().str.contains('CALL')])
    put_strike = _snap_to_strike(put_strike_raw, df[df['Tipo'].str.upper().str.contains('PUT')])

    # Guard: call must be above spot, put below
    if call_strike <= spot:
        call_candidates = df[
            (df['Tipo'].str.upper().str.contains('CALL')) &
            (df['Strike'] > spot)
        ]['Strike'].values
        if len(call_candidates) > 0:
            call_strike = float(call_candidates[np.argmin(np.abs(call_candidates - call_strike_raw))])
        else:
            return None

    if put_strike >= spot:
        put_candidates = df[
            (df['Tipo'].str.upper().str.contains('PUT')) &
            (df['Strike'] < spot)
        ]['Strike'].values
        if len(put_candidates) > 0:
            put_strike = float(put_candidates[np.argmin(np.abs(put_candidates - put_strike_raw))])
        else:
            return None

    # IV lookup
    call_iv = _get_iv(df, call_strike, wk['friday_date'], 'call')
    put_iv = _get_iv(df, put_strike, wk['friday_date'], 'put')

    if np.isnan(call_iv):
        call_iv = _fallback_iv(df, call_strike, 'call')
    if np.isnan(put_iv):
        put_iv = _fallback_iv(df, put_strike, 'put')

    if np.isnan(call_iv) or np.isnan(put_iv):
        return None

    call_price = _bs_price(spot, call_strike, T, r, call_iv, 'call')
    put_price = _bs_price(spot, put_strike, T, r, put_iv, 'put')

    legs = [
        {
            'leg': 'call',
            'action': action,
            'qty': 1,
            'strike': call_strike,
            'expiry': wk['friday_str'],
            'dte': dte,
            'type': 'call',
            'iv': call_iv,
            'price': call_price,
        },
        {
            'leg': 'put',
            'action': action,
            'qty': 1,
            'strike': put_strike,
            'expiry': wk['friday_str'],
            'dte': dte,
            'type': 'put',
            'iv': put_iv,
            'price': put_price,
        },
    ]

    # Net premium: credit for short, debit for long
    sign = 1.0 if action == 'SELL' else -1.0
    net_premium = sign * (call_price + put_price)

    # Aggregate Greeks
    net_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    for leg in legs:
        g = _bs_greeks(spot, leg['strike'], T, r, leg['iv'], leg['type'])
        leg_sign = 1.0 if leg['action'] == 'BUY' else -1.0
        for k in net_greeks:
            net_greeks[k] += leg_sign * leg['qty'] * g[k]

    # P&L profile at expiry
    profile = _pnl_profile(legs, spot, T, r)
    max_profit = float(profile['pnl'].max()) if not profile.empty else np.nan
    max_loss = float(profile['pnl'].min()) if not profile.empty else np.nan
    break_evens = _find_break_evens(profile)

    return {
        'strategy': 'STRANGLE',
        'direction': direction,
        'call_strike': call_strike,
        'put_strike': put_strike,
        'expiry': wk['friday_str'],
        'dte': dte,
        'legs': legs,
        'net_premium': net_premium,
        'net_greeks': net_greeks,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'break_evens': break_evens,
        'pnl_profile': profile,
        'suitability': suitability,
        'rationale': rationale,
        'regime': regime,
    }


def format_strangle_snapshot(result, win_mapper=None):
    """Return a multi-line string with the strangle summary for console output."""
    if result is None:
        return "[STRANGLE] Insufficient data to construct strategy.\n"

    lines = []
    lines.append("=" * 75)
    lines.append(f"STRANGLE STRATEGY ({result['direction']}) -- GEX Walls as Strikes")
    lines.append("=" * 75)
    lines.append(f"  Suitability : {result['suitability']}")
    lines.append(f"  Regime      : {result['regime']}")
    lines.append(f"  Rationale   : {result['rationale']}")
    lines.append("")

    # Legs table
    lines.append(f"  {'LEG':<8} {'ACTION':<6} {'QTY':>4} {'STRIKE':>10} {'EXPIRY':<12} {'DTE':>4} {'IV':>7} {'PRICE':>10}")
    lines.append("  " + "-" * 65)
    for leg in result['legs']:
        win_str = ""
        if win_mapper is not None:
            win_str = f"  (WIN {win_mapper.bova11_to_ind(leg['strike']):,.0f})"
        lines.append(
            f"  {leg['leg']:<8} {leg['action']:<6} {leg['qty']:>4} "
            f"{leg['strike']:>10.2f} {leg['expiry']:<12} {leg['dte']:>4} "
            f"{leg['iv'] * 100:>6.1f}% {leg['price']:>10.4f}"
            f"{win_str}"
        )

    lines.append("")
    net = result['net_premium']
    dc = "CREDIT" if net > 0 else "DEBIT"
    lines.append(f"  Net Premium : {abs(net):.4f} ({dc})")

    if np.isfinite(result['max_profit']):
        mp_label = f"{result['max_profit']:.4f}" if result['direction'] == 'SHORT' else "Unlimited"
        lines.append(f"  Max Profit  : {mp_label}")
    else:
        lines.append("  Max Profit  : N/A")

    if np.isfinite(result['max_loss']):
        ml_label = f"{result['max_loss']:.4f}" if result['direction'] == 'LONG' else f"{result['max_loss']:.4f} (premium collected)"
        lines.append(f"  Max Loss    : {ml_label}")
    else:
        lines.append("  Max Loss    : N/A")

    if result['break_evens']:
        be_str = " / ".join(f"{b:.2f}" for b in result['break_evens'])
        lines.append(f"  Break-Evens : {be_str}")
    else:
        lines.append("  Break-Evens : N/A")

    g = result['net_greeks']
    lines.append("")
    lines.append(f"  Net Delta : {g['delta']:>+8.4f}")
    lines.append(f"  Net Gamma : {g['gamma']:>+8.6f}")
    lines.append(f"  Net Theta : {g['theta']:>+8.4f}  (daily)")
    lines.append(f"  Net Vega  : {g['vega']:>+8.4f}  (per 1% IV)")
    lines.append("=" * 75)

    return "\n".join(lines)

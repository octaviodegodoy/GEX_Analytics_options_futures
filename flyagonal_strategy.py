# -*- coding: utf-8 -*-
"""
Flyagonal (Diagonal Butterfly) Strategy Builder
------------------------------------------------
Constructs diagonal butterfly setups using GEX pin candidates + weekly walls.

Core idea:
  - Short 2x ATM/pin-strike options in the **near** expiry (theta harvest)
  - Long 1x lower + 1x upper wing options in the **far** expiry (vega exposure)
  - Best deployed in positive gamma / low-vol regime near pinning strikes
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes pricing (self-contained for module independence)
# ---------------------------------------------------------------------------

def _bs_price(S, K, T, r, sigma, option_type='call'):
    """European option price. T in years, r as continuous rate."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Return dict with delta, gamma, theta, vega for one leg."""
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
# Flyagonal builder
# ---------------------------------------------------------------------------

def build_flyagonal(df, spot, weekly_results, pin_snapshot, regime,
                    option_type='call', wing_width=None, r=0.10):
    """
    Build a flyagonal (diagonal butterfly) trade from GEX data.

    Parameters
    ----------
    df : pd.DataFrame
        Full options chain with Strike, IV, DTE, Tipo, Tit., Expiration.
    spot : float
        Current underlying price.
    weekly_results : list[dict]
        Output from compute_weekly_walls() — must have >= 2 entries.
    pin_snapshot : dict
        Output from _build_pin_candidates_snapshot().
    regime : str
        Current gamma regime ("POSITIVE GAMMA …", "NEGATIVE GAMMA …", etc.).
    option_type : str
        'call' or 'put' — butterflies can be built with either.
    wing_width : float or None
        Distance from center to wings. Auto-detected from strikes if None.
    r : float
        Annualised risk-free rate (default 10 % for Brazil SELIC-ish).

    Returns
    -------
    dict or None
        Strategy specification with legs, Greeks, P&L profile, and rationale.
        Returns None if insufficient data to construct a valid setup.
    """
    # Need at least 2 expirations
    valid_weeks = [w for w in weekly_results if not w['gex_by_strike'].empty]
    if len(valid_weeks) < 2:
        return None

    near_wk = valid_weeks[0]  # near expiry (sell body)
    far_wk = valid_weeks[1]   # far expiry  (buy wings)

    near_dte = max(near_wk['dte'], 1)
    far_dte = max(far_wk['dte'], 2)
    near_T = near_dte / 252.0
    far_T = far_dte / 252.0

    # --- Select center strike: best pin candidate near spot ---
    pins = pin_snapshot.get('pin_candidates', pd.DataFrame())
    if not pins.empty:
        # Pick pin with highest dealer_gex closest to spot
        pins_sorted = pins.copy()
        pins_sorted['dist'] = (pins_sorted['Strike'] - spot).abs()
        center_strike = float(pins_sorted.sort_values(
            ['dist', 'dealer_gex'], ascending=[True, False]
        ).iloc[0]['Strike'])
    else:
        # Fallback: nearest available strike to spot in near expiry
        near_strikes = near_wk['gex_by_strike']['Strike'].values
        if len(near_strikes) == 0:
            return None
        center_strike = float(near_strikes[np.argmin(np.abs(near_strikes - spot))])

    # --- Wing width: auto-detect from typical strike spacing ---
    if wing_width is None:
        all_strikes = sorted(df['Strike'].unique())
        if len(all_strikes) >= 3:
            diffs = np.diff(all_strikes)
            median_spacing = float(np.median(diffs))
            wing_width = max(median_spacing * 2, spot * 0.01)  # at least 1% of spot
        else:
            wing_width = spot * 0.02

    lower_strike = _snap_to_strike(center_strike - wing_width, df)
    upper_strike = _snap_to_strike(center_strike + wing_width, df)
    center_strike = _snap_to_strike(center_strike, df)

    # Ensure distinct strikes
    if lower_strike >= center_strike or center_strike >= upper_strike:
        return None

    # --- Get IVs from the actual chain ---
    near_iv_center = _get_iv(df, center_strike, near_wk['friday_date'], option_type)
    far_iv_lower = _get_iv(df, lower_strike, far_wk['friday_date'], option_type)
    far_iv_upper = _get_iv(df, upper_strike, far_wk['friday_date'], option_type)

    # Fallback: interpolate from nearby IVs if exact match missing
    if np.isnan(near_iv_center):
        near_iv_center = _fallback_iv(df, center_strike, option_type)
    if np.isnan(far_iv_lower):
        far_iv_lower = _fallback_iv(df, lower_strike, option_type)
    if np.isnan(far_iv_upper):
        far_iv_upper = _fallback_iv(df, upper_strike, option_type)

    if any(np.isnan(v) for v in [near_iv_center, far_iv_lower, far_iv_upper]):
        return None

    # --- Build legs ---
    otype = option_type.lower()
    legs = [
        {
            'leg': 'lower_wing',
            'action': 'BUY',
            'qty': 1,
            'strike': lower_strike,
            'expiry': far_wk['friday_str'],
            'dte': far_dte,
            'type': otype,
            'iv': far_iv_lower,
            'price': _bs_price(spot, lower_strike, far_T, r, far_iv_lower, otype),
        },
        {
            'leg': 'body_1',
            'action': 'SELL',
            'qty': 1,
            'strike': center_strike,
            'expiry': near_wk['friday_str'],
            'dte': near_dte,
            'type': otype,
            'iv': near_iv_center,
            'price': _bs_price(spot, center_strike, near_T, r, near_iv_center, otype),
        },
        {
            'leg': 'body_2',
            'action': 'SELL',
            'qty': 1,
            'strike': center_strike,
            'expiry': near_wk['friday_str'],
            'dte': near_dte,
            'type': otype,
            'iv': near_iv_center,
            'price': _bs_price(spot, center_strike, near_T, r, near_iv_center, otype),
        },
        {
            'leg': 'upper_wing',
            'action': 'BUY',
            'qty': 1,
            'strike': upper_strike,
            'expiry': far_wk['friday_str'],
            'dte': far_dte,
            'type': otype,
            'iv': far_iv_upper,
            'price': _bs_price(spot, upper_strike, far_T, r, far_iv_upper, otype),
        },
    ]

    # --- Net debit / credit ---
    net_premium = (
        - legs[0]['price']   # buy lower wing
        + legs[1]['price']   # sell body
        + legs[2]['price']   # sell body
        - legs[3]['price']   # buy upper wing
    )

    # --- Aggregate Greeks ---
    net_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    for leg in legs:
        T_leg = leg['dte'] / 252.0
        g = _bs_greeks(spot, leg['strike'], T_leg, r, leg['iv'], leg['type'])
        sign = 1.0 if leg['action'] == 'BUY' else -1.0
        for k in net_greeks:
            net_greeks[k] += sign * leg['qty'] * g[k]

    # --- P&L profile at near-term expiry ---
    pnl_profile = _compute_pnl_profile(
        legs, spot, near_T, far_T, r, n_points=50
    )

    # Max profit / loss estimates from profile
    max_profit = float(pnl_profile['pnl'].max()) if not pnl_profile.empty else np.nan
    max_loss = float(pnl_profile['pnl'].min()) if not pnl_profile.empty else np.nan

    # Break-evens
    be = _find_break_evens(pnl_profile)

    # --- Suitability check ---
    is_positive_gamma = 'POSITIVE' in regime.upper()
    suitability = 'IDEAL' if is_positive_gamma else 'SUBOPTIMAL'
    rationale = (
        'Positive gamma regime -- dealers dampen vol, pin effect strong. '
        f'Center strike {center_strike:.2f} has high pin probability.'
    ) if is_positive_gamma else (
        'Negative gamma regime -- vol amplified, pin less reliable. '
        'Consider reducing size or widening wings.'
    )

    return {
        'strategy': 'FLYAGONAL',
        'option_type': otype,
        'center_strike': center_strike,
        'lower_strike': lower_strike,
        'upper_strike': upper_strike,
        'wing_width': wing_width,
        'near_expiry': near_wk['friday_str'],
        'far_expiry': far_wk['friday_str'],
        'near_dte': near_dte,
        'far_dte': far_dte,
        'legs': legs,
        'net_premium': net_premium,
        'net_greeks': net_greeks,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'break_evens': be,
        'pnl_profile': pnl_profile,
        'suitability': suitability,
        'rationale': rationale,
        'regime': regime,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap_to_strike(target, df):
    """Snap a target price to the nearest available strike in the chain."""
    strikes = df['Strike'].unique()
    if len(strikes) == 0:
        return target
    return float(strikes[np.argmin(np.abs(strikes - target))])


def _get_iv(df, strike, expiry_dt, option_type):
    """Retrieve IV from the chain for a specific strike + expiry + type."""
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
    """Get IV from nearest strike of the same type, any expiry."""
    sub = df[df['Tipo'].str.upper().str.contains(option_type.upper())].copy()
    if sub.empty:
        return np.nan
    sub['_dist'] = (sub['Strike'] - strike).abs()
    return float(sub.sort_values('_dist').iloc[0]['IV'])


def _compute_pnl_profile(legs, spot, near_T, far_T, r, n_points=50):
    """
    Estimate P&L at near-term expiry across a range of underlying prices.

    At near-term expiry:
      - Short (near-expiry) legs settle at intrinsic value
      - Long (far-expiry) legs are re-priced with remaining time (far_T - near_T)
    """
    lo = spot * 0.90
    hi = spot * 1.10
    prices = np.linspace(lo, hi, n_points)
    remaining_T = max(far_T - near_T, 1 / 252.0)

    entry_cost = sum(
        (-1.0 if l['action'] == 'BUY' else 1.0) * l['qty'] * l['price']
        for l in legs
    )

    pnl_list = []
    for S in prices:
        value = 0.0
        for leg in legs:
            sign = 1.0 if leg['action'] == 'BUY' else -1.0
            if leg['leg'].startswith('body'):
                # Near-expiry leg: intrinsic at expiry
                if leg['type'] == 'call':
                    val = max(S - leg['strike'], 0.0)
                else:
                    val = max(leg['strike'] - S, 0.0)
            else:
                # Far-expiry leg: BS re-pricing with remaining time
                val = _bs_price(S, leg['strike'], remaining_T, r, leg['iv'], leg['type'])
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
# Console formatter
# ---------------------------------------------------------------------------

def format_flyagonal_snapshot(result, win_mapper=None):
    """Return a multi-line string with the strategy summary for console output."""
    if result is None:
        return "[FLYAGONAL] Insufficient data to construct strategy.\n"

    lines = []
    lines.append("=" * 75)
    lines.append(f"FLYAGONAL STRATEGY (Diagonal Butterfly) -- {result['option_type'].upper()}S")
    lines.append("=" * 75)
    lines.append(f"  Suitability : {result['suitability']}")
    lines.append(f"  Regime      : {result['regime']}")
    lines.append(f"  Rationale   : {result['rationale']}")
    lines.append("")

    # Legs table
    lines.append(f"  {'LEG':<14} {'ACTION':<6} {'QTY':>4} {'STRIKE':>10} {'EXPIRY':<12} {'DTE':>4} {'IV':>7} {'PRICE':>10}")
    lines.append("  " + "-" * 73)
    for leg in result['legs']:
        win_str = ""
        if win_mapper is not None:
            win_str = f"  (WIN {win_mapper.bova11_to_ind(leg['strike']):,.0f})"
        lines.append(
            f"  {leg['leg']:<14} {leg['action']:<6} {leg['qty']:>4} "
            f"{leg['strike']:>10.2f} {leg['expiry']:<12} {leg['dte']:>4} "
            f"{leg['iv'] * 100:>6.1f}% {leg['price']:>10.4f}"
            f"{win_str}"
        )

    lines.append("")
    net = result['net_premium']
    dc = "CREDIT" if net > 0 else "DEBIT"
    lines.append(f"  Net Premium : {abs(net):.4f} ({dc})")
    lines.append(f"  Max Profit  : {result['max_profit']:.4f}" if np.isfinite(result['max_profit']) else "  Max Profit  : N/A")
    lines.append(f"  Max Loss    : {result['max_loss']:.4f}" if np.isfinite(result['max_loss']) else "  Max Loss    : N/A")

    if result['break_evens']:
        be_str = " / ".join(f"{b:.2f}" for b in result['break_evens'])
        lines.append(f"  Break-Evens : {be_str}")
    else:
        lines.append("  Break-Evens : N/A")

    # Net Greeks
    g = result['net_greeks']
    lines.append("")
    lines.append(f"  Net Delta : {g['delta']:>+8.4f}")
    lines.append(f"  Net Gamma : {g['gamma']:>+8.6f}")
    lines.append(f"  Net Theta : {g['theta']:>+8.4f}  (daily)")
    lines.append(f"  Net Vega  : {g['vega']:>+8.4f}  (per 1% IV)")
    lines.append("=" * 75)

    return "\n".join(lines)

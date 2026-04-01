# -*- coding: utf-8 -*-
"""
GEX Utilities — gamma flip detection, weekly wall computation, and related helpers.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _friday_of_week(ref_date: datetime, weeks_ahead: int = 0) -> datetime:
    """Return the Friday of the week ``weeks_ahead`` from *ref_date*."""
    days_until_friday = (4 - ref_date.weekday()) % 7
    friday = ref_date + timedelta(days=days_until_friday)
    friday += timedelta(weeks=weeks_ahead)
    return friday.replace(hour=0, minute=0, second=0, microsecond=0)


def _business_days_between(d1, d2):
    """Count business days (Mon-Fri) between two dates."""
    return max(int(np.busday_count(d1.date() if hasattr(d1, 'date') else d1,
                                   d2.date() if hasattr(d2, 'date') else d2)), 0)


def compute_weekly_walls(df: pd.DataFrame, spot: float):
    """
    Compute gamma walls, gamma flip, and GEX breakdown for the current
    week and the next week expirations.

    Parameters
    ----------
    df : pd.DataFrame
        Full options chain — must already have ``GEX_customer``, ``Tipo``,
        ``Strike``, and ``Expiration`` columns.
    spot : float
        Current underlying spot price.

    Returns
    -------
    list[dict]
        One dict per week with keys:
            label, friday_date, friday_str, dte, calls, puts,
            gex_by_strike, total_gex, peak_gex_strike,
            gamma_flip, call_wall, put_wall
        The list is empty if no options match either week.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df = df.copy()
    df['Expiration'] = pd.to_datetime(df['Expiration'])

    # Determine the two target Fridays
    current_friday = _friday_of_week(today)
    if current_friday.date() < today.date():
        # Friday already passed this week (Sat/Sun) — shift to next week
        current_friday = _friday_of_week(today, weeks_ahead=1)
    next_friday = current_friday + timedelta(weeks=1)

    # All available expiration dates, sorted
    all_expirations = sorted(df['Expiration'].dt.date.unique())

    weeks = [
        ("Current Week", current_friday),
        ("Next Week",    next_friday),
    ]

    used_dates = set()
    results = []
    for label, friday in weeks:
        target_date = friday.date()
        wk_df = df[df['Expiration'].dt.date == target_date]

        # If no options on this Friday, pick the next available expiration
        if wk_df.empty:
            candidates = [d for d in all_expirations
                          if d > (current_friday.date() if label == "Next Week" else today.date())
                          and d not in used_dates]
            if candidates:
                target_date = candidates[0]
                friday = datetime.combine(target_date, datetime.min.time())
                wk_df = df[df['Expiration'].dt.date == target_date]

        if wk_df.empty:
            results.append({
                'label': label,
                'friday_date': friday,
                'friday_str': friday.strftime('%Y-%m-%d'),
                'dte': _business_days_between(today, friday),
                'calls': pd.DataFrame(),
                'puts': pd.DataFrame(),
                'gex_by_strike': pd.DataFrame(),
                'total_gex': 0.0,
                'peak_gex_strike': np.nan,
                'gamma_flip': np.nan,
                'call_wall': np.nan,
                'put_wall': np.nan,
            })
            continue

        used_dates.add(target_date)

        wk_calls = wk_df[wk_df['Tipo'].str.upper().str.contains('CALL')]
        wk_puts  = wk_df[wk_df['Tipo'].str.upper().str.contains('PUT')]

        gex_by_strike = wk_df.groupby('Strike', as_index=False).agg(
            GEX_customer=('GEX_customer', 'sum')
        ).sort_values('Strike')

        total_gex = gex_by_strike['GEX_customer'].sum()
        peak_idx = gex_by_strike['GEX_customer'].abs().idxmax()
        peak_gex_strike = gex_by_strike.loc[peak_idx, 'Strike']

        gamma_flip = find_gamma_flip(wk_df, spot)

        # Call wall: strike with max call GEX >= spot
        call_gex = wk_calls.groupby('Strike')['GEX_customer'].sum() if not wk_calls.empty else pd.Series(dtype=float)
        call_above = call_gex[call_gex.index >= spot]
        call_wall = call_above.idxmax() if not call_above.empty else np.nan

        # Put wall: strike with max |put GEX| <= spot
        put_gex = wk_puts.groupby('Strike')['GEX_customer'].sum() if not wk_puts.empty else pd.Series(dtype=float)
        put_below = put_gex[put_gex.index <= spot]
        put_wall = put_below.abs().idxmax() if not put_below.empty else np.nan

        results.append({
            'label': label,
            'friday_date': friday,
            'friday_str': friday.strftime('%Y-%m-%d'),
            'dte': _business_days_between(today, friday),
            'calls': wk_calls,
            'puts': wk_puts,
            'gex_by_strike': gex_by_strike,
            'total_gex': total_gex,
            'peak_gex_strike': peak_gex_strike,
            'gamma_flip': gamma_flip,
            'call_wall': call_wall,
            'put_wall': put_wall,
        })

    return results


def find_gamma_flip(df_options, spot, grid_step=0.25, pct_range=0.15):
    """
    Scan-based Gamma Flip — find the price where net customer GEX crosses zero.

    For each candidate price S on a fine grid around *spot*, re-evaluates
    Black-Scholes gamma for every option using its own IV, then sums the
    signed GEX (calls +, puts −).  The flip is the S where that sum
    crosses zero nearest to spot.

    This is more accurate than the per-strike approach because deep-OTM
    positions naturally fade when the test price moves away from their strike.

    Parameters
    ----------
    df_options : pd.DataFrame
        Raw option rows — must contain ``Strike``, ``IV``, ``DTE``,
        ``Tipo`` (CALL / PUT), and ``Tit.`` (open interest / volume proxy).
    spot : float
        Current underlying price.
    grid_step : float
        Price grid resolution (default 0.25).
    pct_range : float
        Fraction of spot for the scan window (default 0.15 = ±15 %).

    Returns
    -------
    float
        Gamma flip price, or ``np.nan`` if it cannot be determined.
    """
    df = df_options.copy()
    df = df[(df['DTE'] > 0) & (df['Strike'] > 0) & (df['Tit.'] > 0)]
    if df.empty or spot <= 0:
        return np.nan

    K   = df['Strike'].to_numpy(dtype=float)
    tau = (df['DTE'].to_numpy(dtype=float)) / 252.0
    oi  = df['Tit.'].to_numpy(dtype=float)
    iv  = df['IV'].to_numpy(dtype=float)
    sign = np.where(df['Tipo'].str.upper().str.contains('PUT'), -1.0, 1.0)

    # Replace zero / nan IV with a conservative fallback
    iv = np.where((iv > 0) & np.isfinite(iv), iv, 0.30)

    lo = spot * (1.0 - pct_range)
    hi = spot * (1.0 + pct_range)
    price_grid = np.arange(lo, hi + grid_step, grid_step)

    net_gex = np.empty(len(price_grid))
    for i, S in enumerate(price_grid):
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + 0.5 * iv**2 * tau) / (iv * np.sqrt(tau))
            pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
            gamma = np.where(np.isfinite(d1), pdf_d1 / (S * iv * np.sqrt(tau)), 0.0)
        gex_i = gamma * oi * (S ** 2) * sign
        net_gex[i] = float(np.nansum(gex_i))

    # Detect zero crossings
    s = np.sign(net_gex)
    s[s == 0] = 1
    crossings = np.where(np.diff(s) != 0)[0]

    if len(crossings) > 0:
        # Linear interpolation between bracketing grid points
        flips = []
        for idx in crossings:
            g0, g1 = net_gex[idx], net_gex[idx + 1]
            p0, p1 = price_grid[idx], price_grid[idx + 1]
            flip = p0 + (0 - g0) * (p1 - p0) / (g1 - g0) if g1 != g0 else p0
            flips.append(flip)
        return float(min(flips, key=lambda f: abs(f - spot)))

    # No crossing — return price with smallest |net GEX|
    return float(price_grid[np.argmin(np.abs(net_gex))])

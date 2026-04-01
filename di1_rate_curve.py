# -*- coding: utf-8 -*-
"""
DI1 Interest-Rate Curve
-----------------------
Fetches the DI1 futures strip from MT5, builds a cubic-spline term
structure, and returns an annualised rate for any target date.

Falls back to a flat SELIC rate when DI1 data is unavailable.
"""
import numpy as np
import pandas as pd
from datetime import datetime

FALLBACK_RATE = 0.1425  # flat SELIC fallback

# ---------------------------------------------------------------------------
# Cubic spline (natural boundary conditions) — self-contained, no extra deps
# ---------------------------------------------------------------------------

def _c_spline(x_data, y_data, x_eval):
    """Cubic spline interpolation with natural boundary conditions."""
    xin = np.array(x_data, dtype=float)
    yin = np.array(y_data, dtype=float)
    if len(xin) != len(yin):
        raise ValueError("x_data and y_data must have the same length!")
    if len(xin) < 2:
        raise ValueError("Need at least 2 data points for interpolation!")

    n = len(xin)
    u = np.zeros(n)
    yt = np.zeros(n)

    for i in range(1, n - 1):
        sig = (xin[i] - xin[i - 1]) / (xin[i + 1] - xin[i - 1])
        p = sig * yt[i - 1] + 2.0
        yt[i] = (sig - 1.0) / p
        u[i] = ((yin[i + 1] - yin[i]) / (xin[i + 1] - xin[i]) -
                (yin[i] - yin[i - 1]) / (xin[i] - xin[i - 1]))
        u[i] = (6.0 * u[i] / (xin[i + 1] - xin[i - 1]) - sig * u[i - 1]) / p

    yt[n - 1] = 0.0
    for k in range(n - 2, -1, -1):
        yt[k] = yt[k] * yt[k + 1] + u[k]

    klo, khi = 0, n - 1
    while khi - klo > 1:
        k = (khi + klo) // 2
        if xin[k] > x_eval:
            khi = k
        else:
            klo = k

    h = xin[khi] - xin[klo]
    if h == 0.0:
        raise ValueError("Duplicate x values in input data!")
    a = (xin[khi] - x_eval) / h
    b = (x_eval - xin[klo]) / h
    return (a * yin[klo] + b * yin[khi] +
            ((a**3 - a) * yt[klo] + (b**3 - b) * yt[khi]) * h**2 / 6.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_cached_curve = None  # (time_numeric, rates, base_time)


def build_di1_curve(mt5_conn):
    """Fetch DI1 monthly closes from MT5 and cache the curve arrays.

    Returns True if data was loaded, False on failure (fallback will be used).
    """
    global _cached_curve
    try:
        data = mt5_conn.get_data("DI1$", mt5_conn.TIMEFRAME_D1, 120, 0)
        if data is None or data.empty:
            print("[DI1] No DI1$ data from MT5 — using flat SELIC fallback")
            _cached_curve = None
            return False

        data = data.sort_values('time', ascending=True).drop_duplicates(subset='time')
        base_time = data['time'].iloc[0]
        time_numeric = (data['time'] - base_time).dt.total_seconds() / 86400.0
        rates = data['close'].values / 100.0  # DI1 prices are in % -> decimal

        _cached_curve = (time_numeric.values, rates, base_time)
        print(f"[DI1] Curve built: {len(rates)} points, range "
              f"{data['time'].iloc[0].date()} -> {data['time'].iloc[-1].date()}")
        return True
    except Exception as e:
        print(f"[DI1] Failed to build curve: {e} -- using flat SELIC fallback")
        _cached_curve = None
        return False


def get_rate_for_date(target_date):
    """Return the annualised rate for *target_date* via spline interpolation.

    If the DI1 curve was not loaded, returns the flat SELIC fallback.
    """
    if _cached_curve is None:
        return FALLBACK_RATE

    time_numeric, rates, base_time = _cached_curve

    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    if hasattr(target_date, 'to_pydatetime'):
        target_date = target_date.to_pydatetime()

    target_numeric = (target_date - base_time).total_seconds() / 86400.0

    # Clamp to the curve's range to avoid extrapolation blow-ups
    target_numeric = max(time_numeric[0], min(time_numeric[-1], target_numeric))

    try:
        return float(_c_spline(time_numeric, rates, target_numeric))
    except Exception:
        return FALLBACK_RATE

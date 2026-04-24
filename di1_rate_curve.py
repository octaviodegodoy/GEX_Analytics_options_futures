# -*- coding: utf-8 -*-
"""
DI1 Interest-Rate Curve
-----------------------
Fetches the risk-free curve from external or market sources, builds a
cubic-spline term structure, and returns an annualised rate for any
target date.

Falls back to a flat SELIC rate when DI1 data is unavailable.
"""
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import BytesIO
import zipfile

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
_BCB_SERIES = (
    (1178, 'Selic over annualizada'),
    (432, 'Meta Selic'),
    (11, 'Selic diaria'),
)

_FUT_MONTH_CODE = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12,
}


def _get_recent_business_days(max_attempts=10):
    days = []
    d = datetime.now()
    while len(days) < max_attempts:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return days


def _fetch_b3_cotahist_text(day):
    date_str = day.strftime('%d%m%Y')
    url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D{date_str}.ZIP"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        txt_names = [n for n in zf.namelist() if n.upper().endswith('.TXT')]
        if not txt_names:
            return None
        with zf.open(txt_names[0]) as f:
            return f.read().decode('latin-1', errors='ignore')


def _parse_maturity_from_ticker(ticker, expiration):
    # Prefer COTAHIST expiration when available.
    if expiration and expiration not in {'00000000', '99991231'}:
        try:
            return datetime.strptime(expiration, '%Y%m%d')
        except ValueError:
            pass

    # Fallback: DI1 month-code + YY, e.g. DI1F27.
    if len(ticker) >= 6 and ticker.startswith('DI1'):
        month = _FUT_MONTH_CODE.get(ticker[3].upper())
        yy = ticker[4:6]
        if month and yy.isdigit():
            year = 2000 + int(yy)
            return datetime(year, month, 1)
    return None


def _rate_from_close(close, quote_date, maturity):
    if close is None or not np.isfinite(close) or close <= 0:
        return None

    # DI1 in many feeds is already in percent (e.g., 13.45).
    if close <= 100.0:
        rate = close / 100.0 if close > 2.0 else close
        return rate if -0.05 <= rate <= 1.5 else None

    # On B3, DI1 may come as PU (around 100000).
    if maturity is None:
        return None

    du = int(np.busday_count(quote_date.date(), maturity.date()))
    if du <= 0:
        return None

    rate = (100000.0 / close) ** (252.0 / du) - 1.0
    return rate if -0.05 <= rate <= 1.5 else None


def _build_curve_from_mt5(mt5_conn):
    candidates = [
        ('DI1$', mt5_conn.TIMEFRAME_D1, 120),
        ('DI1@', mt5_conn.TIMEFRAME_D1, 120),
        ('DI1@', mt5_conn.TIMEFRAME_MN1, 24),
        ('DI1', mt5_conn.TIMEFRAME_D1, 120),
    ]

    for symbol, timeframe, bars in candidates:
        try:
            data = mt5_conn.get_data(symbol, timeframe, bars, 0)
        except Exception:
            data = None

        if data is None or data.empty or 'time' not in data.columns or 'close' not in data.columns:
            continue

        d = data.sort_values('time', ascending=True).drop_duplicates(subset='time').copy()
        d['time'] = pd.to_datetime(d['time'])
        d['rate'] = d['close'].apply(lambda x: _rate_from_close(float(x), d['time'].iloc[-1], None))
        d = d.dropna(subset=['rate'])
        if len(d) < 2:
            continue

        base_time = d['time'].iloc[0].to_pydatetime()
        time_numeric = (d['time'] - d['time'].iloc[0]).dt.total_seconds() / 86400.0
        rates = d['rate'].values.astype(float)
        print(f"[DI1] Curve built from MT5 {symbol}: {len(rates)} points")
        return (time_numeric.values, rates, base_time)

    return None


def _build_curve_from_bcb():
    """Build a flat risk-free curve from Banco Central SGS series."""
    for series_id, label in _BCB_SERIES:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados/ultimos/5?formato=json"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            continue

        if not payload:
            continue

        for row in reversed(payload):
            try:
                quote_date = datetime.strptime(row['data'], '%d/%m/%Y')
                raw_value = float(str(row['valor']).replace(',', '.'))
            except Exception:
                continue

            if series_id == 11:
                rate = (1.0 + raw_value / 100.0) ** 252 - 1.0
            else:
                rate = raw_value / 100.0

            if not np.isfinite(rate) or not (-0.05 <= rate <= 1.5):
                continue

            horizon_days = np.array([0.0, 3650.0], dtype=float)
            rates = np.array([rate, rate], dtype=float)
            print(f"[DI1] Curve built from BCB SGS {series_id} ({label}): {rate * 100:.2f}%")
            return (horizon_days, rates, quote_date)

    return None


def _build_curve_from_b3(max_attempts=7):
    records = []

    for day in _get_recent_business_days(max_attempts=max_attempts):
        try:
            content = _fetch_b3_cotahist_text(day)
        except Exception:
            content = None

        if not content:
            continue

        for line in content.splitlines():
            if len(line) < 210 or line[0:2] != '01':
                continue

            ticker = line[12:24].strip()
            if not ticker.startswith('DI1'):
                continue

            quote_raw = line[2:10].strip()
            exp_raw = line[202:210].strip()
            close_raw = line[108:121].strip()

            try:
                quote_date = datetime.strptime(quote_raw, '%Y%m%d')
                close = int(close_raw) / 100.0
            except Exception:
                continue

            maturity = _parse_maturity_from_ticker(ticker, exp_raw)
            if maturity is None or maturity <= quote_date:
                continue

            rate = _rate_from_close(close, quote_date, maturity)
            if rate is None:
                continue

            du = int(np.busday_count(quote_date.date(), maturity.date()))
            if du <= 0:
                continue

            records.append((quote_date, ticker, du, rate))

        if records:
            break

    if not records:
        return None

    # Use only the freshest quote day available.
    latest_qd = max(r[0] for r in records)
    latest = [r for r in records if r[0] == latest_qd]

    df = pd.DataFrame(latest, columns=['quote_date', 'ticker', 'du', 'rate'])
    # Keep one point per maturity bucket.
    df = df.groupby('du', as_index=False)['rate'].mean().sort_values('du')
    if len(df) < 2:
        return None

    print(f"[DI1] Curve built from B3 COTAHIST: {len(df)} maturities ({latest_qd.date()})")
    return (df['du'].values.astype(float), df['rate'].values.astype(float), latest_qd)


def build_di1_curve(mt5_conn):
    """Fetch the risk-free curve and cache the curve arrays.

    Returns True if data was loaded, False on failure (fallback will be used).
    """
    global _cached_curve
    try:
        bcb_curve = _build_curve_from_bcb()
        if bcb_curve is not None:
            _cached_curve = bcb_curve
            return True

        mt5_curve = _build_curve_from_mt5(mt5_conn)
        if mt5_curve is not None:
            _cached_curve = mt5_curve
            return True

        print("[DI1] External BCB source unavailable. Trying MT5/B3 fallbacks...")
        b3_curve = _build_curve_from_b3(max_attempts=7)
        if b3_curve is not None:
            _cached_curve = b3_curve
            return True

        print("[DI1] No external/market rate data — using flat SELIC fallback")
        _cached_curve = None
        return False
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

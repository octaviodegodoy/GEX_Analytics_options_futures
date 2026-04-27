# -*- coding: utf-8 -*-
"""
RTD OI reader — CSV only.

Reads Open Interest (and optional strike) data from the snapshot CSV produced
by Excel + ``export_rtd_oi_from_excel.bat`` (or any equivalent exporter):

    <MQL5_root>/Files/RTD_OI.csv

This module deliberately does NOT talk to Profit Pro's RTD COM server.
Profit's RTDTrading.RTDServer is unstable: subscribing to an unknown / expired
option ticker can crash ``profitchart.exe`` with an access violation and
poison every subsequent COM call in the same Python process. The CSV path is
the single source of truth here.

Public API
----------
- ``read_rtd_oi(filepath=None, spot=None, strikes_around=15, ...)``
    DataFrame with columns ``ticker``, ``oi`` (and ``strike`` when present).
- ``read_rtd_option_snapshot(tickers, attributes=['CAB'], ...)``
    DataFrame with one row per ticker and one column per requested attribute
    (lower-cased): ``cab`` (OI) and ``pex`` (strike) come from the CSV.
    ``ult`` (last price) is not in the CSV and is returned as NaN.
- ``rtd_data_changed()`` — True when the CSV mtime advanced since last call.
- ``rtd_file_mtime(filepath=None)`` — mtime of the CSV (0.0 if missing).
- ``rtd_shutdown()`` — no-op kept for backwards compatibility.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import time
import unicodedata

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MQL5_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

#: Default RTD CSV path written by the Excel exporter.
RTD_OI_PATH = os.path.join(MQL5_ROOT, 'Files', 'RTD_OI.csv')

#: Maximum age (seconds) for the RTD CSV to be considered current-day data.
#: RTD readings are only valid for the current trading day; a stale file from a
#: previous session must not silently feed the GEX calculation.
RTD_CSV_MAX_AGE_SECONDS = 30 * 60  # 30 minutes


# ----------------------------------------------------------------------------
# Column-name aliases — Profit Pro / Excel exports use Portuguese headers.
# ----------------------------------------------------------------------------
_TICKER_ALIASES = {
    'codigo', 'codneg', 'ticker', 'symbol', 'asset', 'ativo', 'serie',
    'cod', 'instrumento', 'opcao',
}
_OI_ALIASES = {
    'qtd.aberta', 'qtd_aberta', 'qtdaberta', 'posição', 'posicao',
    'contratos_abertos', 'open_interest', 'openinterest', 'oi',
    'pos.aberta', 'pos_aberta', 'posaberta', 'contratos',
    'cont.abertos', 'cont_abertos', 'contabertos', 'cab',
}
_STRIKE_ALIASES = {
    'strike', 'exercicio', 'preco_exercicio', 'preço_exercício',
    'preco', 'pe', 'pex',
}
_TYPE_ALIASES = {
    'tipo', 'type', 'call_put', 'c/p', 'cp', 'natureza',
}


def _normalize_colname(name: str) -> str:
    """Normalize CSV headers across locale/punctuation variants."""
    text = str(name).strip().lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


# ----------------------------------------------------------------------------
# Freshness check
# ----------------------------------------------------------------------------
def _is_csv_fresh(path: str, max_age_seconds: int = RTD_CSV_MAX_AGE_SECONDS) -> bool:
    """Return True if the CSV mtime is from today and within max_age_seconds."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return False
    now = time.time()
    age = now - mtime
    today = _dt.date.today()
    file_date = _dt.date.fromtimestamp(mtime)
    if file_date != today:
        print(f"[RTD CSV] Rejecting stale file: mtime date {file_date} != today {today} "
              f"({path})")
        return False
    if age > max_age_seconds:
        print(f"[RTD CSV] Rejecting stale file: age {age:.0f}s > "
              f"max {max_age_seconds}s ({path})")
        return False
    return True


def rtd_file_mtime(filepath: str = None) -> float:
    """Return the CSV file modification timestamp, or 0.0 if not found."""
    path = filepath or RTD_OI_PATH
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


# ----------------------------------------------------------------------------
# Core CSV loader
# ----------------------------------------------------------------------------
def _load_csv_raw(path: str) -> pd.DataFrame:
    """Read the CSV with header normalisation. Returns canonical columns:
    ticker, oi, strike (if present), type (if present)."""
    df = None
    for enc in ('utf-8-sig', 'latin-1', 'cp1252'):
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine='python')
            break
        except Exception:
            continue

    if df is None or df.empty:
        print(f"[RTD CSV] Could not read {path}")
        return pd.DataFrame()

    col_map = {}
    for c in df.columns:
        cl = _normalize_colname(c)
        if cl in _TICKER_ALIASES:
            col_map[c] = 'ticker'
        elif cl in _OI_ALIASES:
            col_map[c] = 'oi'
        elif cl in _STRIKE_ALIASES:
            col_map[c] = 'strike'
        elif cl in _TYPE_ALIASES:
            col_map[c] = 'type'

    df = df.rename(columns=col_map)

    if 'ticker' not in df.columns or 'oi' not in df.columns:
        print(f"[RTD CSV] Missing required columns. Found: {list(df.columns)}")
        print(f"[RTD CSV] Need 'ticker' + 'oi' (or Profit Pro equivalents)")
        return pd.DataFrame()

    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
    df['oi'] = pd.to_numeric(df['oi'], errors='coerce').fillna(0)
    if 'strike' in df.columns:
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')

    return df


def _read_csv_oi(filepath: str = None, spot: float = None,
                 strikes_around: int = 15,
                 enforce_freshness: bool = True,
                 max_age_seconds: int = RTD_CSV_MAX_AGE_SECONDS) -> pd.DataFrame:
    """Read OI from the RTD CSV. Filters to OI > 0, and (optionally) to
    ``strikes_around`` strikes on each side of ``spot``.

    By default, files older than ``max_age_seconds`` or dated before today are
    rejected to prevent stale OI from a previous session leaking into the GEX
    parameters. Pass ``enforce_freshness=False`` for backtest/inspection only.
    """
    path = filepath or RTD_OI_PATH

    if not os.path.exists(path):
        return pd.DataFrame()

    if enforce_freshness and not _is_csv_fresh(path, max_age_seconds=max_age_seconds):
        return pd.DataFrame()

    df = _load_csv_raw(path)
    if df.empty:
        return df

    df = df[df['oi'] > 0].copy()

    # Filter to ±strikes_around strikes around spot
    if spot is not None and spot > 0 and 'strike' in df.columns:
        side = df.dropna(subset=['strike'])
        unique_strikes = sorted(side['strike'].unique())
        if unique_strikes:
            import bisect
            idx = bisect.bisect_left(unique_strikes, spot)
            lo = max(0, idx - strikes_around)
            hi = min(len(unique_strikes), idx + strikes_around)
            keep = set(unique_strikes[lo:hi])
            before = len(df)
            df = df[df['strike'].isin(keep) | df['strike'].isna()]
            print(f"[RTD CSV] Filtered to {len(keep)} strikes around spot "
                  f"{spot:.2f} ({before} -> {len(df)} rows)")

    cols = ['ticker', 'oi'] + (['strike'] if 'strike' in df.columns else [])
    return df[cols]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def read_rtd_oi(filepath: str = None, spot: float = None,
                strikes_around: int = 15,
                tickers: list = None,
                enforce_freshness: bool = True,
                max_age_seconds: int = RTD_CSV_MAX_AGE_SECONDS) -> pd.DataFrame:
    """Get OI data from the RTD CSV snapshot.

    Parameters
    ----------
    filepath : str, optional
        CSV path (defaults to ``RTD_OI_PATH``).
    spot : float, optional
        Current spot price for strike filtering.
    strikes_around : int
        Strikes to keep on each side of spot (default 15).
    tickers : list of str, optional
        If given, restrict the result to these tickers (case-insensitive).
        Kept for backwards compatibility — has no other effect now that the
        COM RTD path has been removed.
    enforce_freshness : bool
        Reject CSV files dated before today or older than ``max_age_seconds``.
    max_age_seconds : int
        Maximum CSV file age accepted when ``enforce_freshness`` is True.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker`` (str), ``oi`` (float), and ``strike`` if present.
    """
    df = _read_csv_oi(filepath=filepath, spot=spot,
                      strikes_around=strikes_around,
                      enforce_freshness=enforce_freshness,
                      max_age_seconds=max_age_seconds)

    if tickers and not df.empty:
        wanted = {str(t).strip().upper() for t in tickers if str(t).strip()}
        if wanted:
            df = df[df['ticker'].isin(wanted)].copy()

    return df


def read_rtd_option_snapshot(tickers: list,
                             attributes: list = None,
                             wait_seconds: float = 1.0,
                             refresh_rounds: int = 3) -> pd.DataFrame:
    """Snapshot of CSV-derived attributes for a list of option tickers.

    Supported attributes (case-insensitive):
      - ``CAB`` -> open interest (from CSV ``oi`` column)
      - ``PEX`` -> strike (from CSV ``strike`` column, if present)
      - ``ULT`` -> last price (NOT in CSV -> returned as NaN)

    ``wait_seconds`` and ``refresh_rounds`` are accepted for backwards
    compatibility and ignored: the CSV is read once.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker`` plus one lowercase column per requested attribute.
        Empty DataFrame if no tickers given or CSV unavailable/stale.
    """
    if not tickers:
        return pd.DataFrame()

    attrs = [str(a).strip().upper() for a in (attributes or ['CAB']) if str(a).strip()]
    if not attrs:
        attrs = ['CAB']

    wanted = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not wanted:
        return pd.DataFrame()

    path = RTD_OI_PATH
    if not os.path.exists(path):
        return pd.DataFrame()
    if not _is_csv_fresh(path):
        return pd.DataFrame()

    raw = _load_csv_raw(path)
    if raw.empty:
        return pd.DataFrame()

    # Drop duplicate tickers (keep first) and index for fast lookup.
    raw = raw.drop_duplicates(subset=['ticker'], keep='first')
    raw_idx = raw.set_index('ticker')

    has_strike = 'strike' in raw_idx.columns
    rows = []
    for tk in wanted:
        row = {'ticker': tk}
        src = raw_idx.loc[tk] if tk in raw_idx.index else None
        for attr in attrs:
            key = attr.lower()
            if src is None:
                row[key] = float('nan')
                continue
            if attr == 'CAB':
                try:
                    row[key] = float(src['oi'])
                except (ValueError, TypeError, KeyError):
                    row[key] = float('nan')
            elif attr == 'PEX':
                if not has_strike:
                    row[key] = float('nan')
                    continue
                try:
                    row[key] = float(src['strike'])
                except (ValueError, TypeError, KeyError):
                    row[key] = float('nan')
            else:
                # Anything else (including ULT) is not in the CSV.
                row[key] = float('nan')
        rows.append(row)

    return pd.DataFrame(rows)


_rtd_last_seen = 0.0


def rtd_data_changed() -> bool:
    """Return True if the RTD CSV's mtime advanced since the last call."""
    global _rtd_last_seen
    mtime = rtd_file_mtime()
    if mtime > _rtd_last_seen:
        _rtd_last_seen = mtime
        return True
    return False


def rtd_shutdown():
    """No-op. Kept for backwards compatibility with old COM-based callers."""
    return None

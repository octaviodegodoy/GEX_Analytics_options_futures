# -*- coding: utf-8 -*-
"""
B3 Options Data Loader
----------------------
Fetches B3 COTAHIST data, classifies call/put, computes Greeks via
Black-Scholes, and merges real OI when available.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from bs_greeks import (
    bs_gamma,
    bs_delta,
    implied_vol,
)
from di1_rate_curve import get_rate_for_date, FALLBACK_RATE

# Resolve paths so get_b3_data is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)

from get_b3_data import fetch_b3_historical_file, fetch_open_interest, search_b3_historical_file


def load_b3_options_data(underlying, spot, date=None):
    """
    Fetch options from B3 historical file and compute Greeks via Black-Scholes.
    Returns DataFrame with columns:
        Ticker, Tipo, Strike, Ultimo, IV, Delta, Gamma, Tit., VolFin
    """
    raw = fetch_b3_historical_file(date)
    if raw.empty:
        print("[*] Primary fetch returned no data — searching recent business days...")
        raw = search_b3_historical_file(max_attempts=7)
    if raw.empty:
        return pd.DataFrame()

    prefix = underlying[:4].upper()
    call_letters = set('ABCDEFGHIJKL')
    put_letters = set('MNOPQRSTUVWX')

    options = raw[raw['ticker'].str.startswith(prefix, na=False)].copy()
    if options.empty:
        print(f"No options found for {underlying}")
        return pd.DataFrame()

    # Classify call/put from the series letter (5th char of ticker)
    def classify_type(ticker):
        if len(ticker) > 4:
            letter = ticker[4].upper()
            if letter in call_letters:
                return 'CALL'
            elif letter in put_letters:
                return 'PUT'
        return None

    options['Tipo'] = options['ticker'].apply(classify_type)
    options = options.dropna(subset=['Tipo'])

    # Parse expiration and compute time to expiry + DTE (business days)
    now = datetime.now()
    def parse_expiration(exp_str):
        try:
            exp_date = datetime.strptime(str(exp_str).strip(), '%Y%m%d')
            dte = max(int(np.busday_count(now.date(), exp_date.date())), 0)
            T = max(dte / 252.0, 1 / 252)
            return T, dte, exp_date
        except (ValueError, TypeError):
            return 20 / 252, 20, now  # fallback ~1 month bdays

    parsed = options['expiration'].apply(parse_expiration)
    options['T'] = parsed.apply(lambda x: x[0])
    options['DTE'] = parsed.apply(lambda x: x[1])
    options['Expiration'] = parsed.apply(lambda x: x[2])

    ivs, gammas, deltas, rates = [], [], [], []
    for _, row in options.iterrows():
        opt_type = row['Tipo'].lower()
        strike = float(row['strike'])
        close = float(row['close'])
        T = float(row['T'])
        exp_date = row['Expiration']

        # Per-expiry rate from DI1 spline (falls back to flat SELIC)
        r = get_rate_for_date(exp_date)

        # Implied vol from market price
        if close > 0 and strike > 0:
            iv = implied_vol(close, spot, strike, T, r, opt_type)
        else:
            iv = 0.30

        gammas.append(bs_gamma(spot, strike, T, r, iv))
        deltas.append(bs_delta(spot, strike, T, r, iv, opt_type))
        ivs.append(iv)  # stored as decimal
        rates.append(r)

    df = pd.DataFrame({
        'Ticker': options['ticker'].values,
        'Tipo': options['Tipo'].values,
        'Strike': options['strike'].values.astype(float),
        'Ultimo': options['close'].values.astype(float),
        'IV': np.array(ivs),
        'Delta': np.array(deltas),
        'Gamma': np.array(gammas),
        'Tit.': options['quantity'].values.astype(float),
        'VolFin': options['volume'].values.astype(float),
        'DTE': options['DTE'].values.astype(int),
        'Expiration': options['Expiration'].values,
    })

    # ---- Merge real OI data if available ----
    oi_source = 'daily_volume'
    try:
        oi_data = fetch_open_interest(
            underlying=underlying,
            multiday_days=5,
        )
        if not oi_data.empty and 'ticker' in oi_data.columns and 'oi' in oi_data.columns:
            oi_map = oi_data.set_index('ticker')['oi'].to_dict()
            oi_source = oi_data['oi_source'].iloc[0] if 'oi_source' in oi_data.columns else 'external'

            # Replace Tit. with real OI where available
            matched = 0
            for idx, row in df.iterrows():
                ticker = row['Ticker']
                if ticker in oi_map and oi_map[ticker] > 0:
                    df.at[idx, 'Tit.'] = oi_map[ticker]
                    matched += 1

            print(f"[*] OI source: {oi_source} | Matched {matched}/{len(df)} options")
        else:
            print("[*] OI source: daily_volume (no better source available)")
    except Exception as e:
        print(f"[*] OI source: daily_volume (fetch error: {e})")

    calls_n = (df['Tipo'] == 'CALL').sum()
    puts_n = (df['Tipo'] == 'PUT').sum()
    dte_dist = df['DTE'].value_counts().sort_index()
    print(f"[*] Built chain: {len(df)} records ({calls_n} calls, {puts_n} puts)")
    print(f"   DTE distribution: {dict(dte_dist.head(5))}")
    print(f"   GEX weight: {'OI' if 'volume' not in oi_source else 'Volume (OI proxy)'}")

    # ---- Filter to only the 2 next expiring dates for GEX analysis ----
    today = pd.Timestamp.now().normalize()
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    all_expirations = sorted(df['Expiration'].dt.normalize().unique())
    future_expirations = [d for d in all_expirations if d >= today]
    if not future_expirations:
        future_expirations = all_expirations[-2:] if len(all_expirations) >= 2 else all_expirations

    next_2 = future_expirations[:2]

    if next_2:
        df = df[df['Expiration'].dt.normalize().isin(next_2)].copy()

    print(f"\n[*] 2 NEXT EXPIRING DATES for GEX analysis:")
    for i, exp in enumerate(next_2, 1):
        exp_ts = pd.Timestamp(exp)
        dte = max(int(np.busday_count(today.date(), exp_ts.date())), 0)
        n_opts = len(df[df['Expiration'].dt.normalize() == exp_ts])
        n_calls = len(df[(df['Expiration'].dt.normalize() == exp_ts) & (df['Tipo'] == 'CALL')])
        n_puts = len(df[(df['Expiration'].dt.normalize() == exp_ts) & (df['Tipo'] == 'PUT')])
        print(f"   {i}. {exp_ts.strftime('%Y-%m-%d')} ({dte} BD) — {n_opts} contracts ({n_calls}C / {n_puts}P)")
    if len(next_2) == 0:
        print("   No future expirations found in data!")

    total_after = len(df)
    print(f"[*] Filtered chain: {total_after} records (kept 2 nearest expirations only)")

    return df

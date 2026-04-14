# -*- coding: utf-8 -*-
"""
Analyze optimal trading hours by sweeping GEX_TRADE_WINDOW_START/END
across last week's cached data, reusing the backtest simulation engine.
"""
import os, sys, importlib
import numpy as np
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import constants
import backtest_dca_gex as bt
from backtest_dca_gex import (
    _compute_gex_walls_for_day, simulate_day, align_tick,
    MARGIN_BUDGET, FIB_SEQ, FIB_TOTAL, FREE_MARGIN,
    MARGIN_PER_LOT, TICK_SIZE, PNL_PER_POINT,
)

def run_sweep():
    import MetaTrader5 as mt5
    from mt5_connector import MT5Connector
    from di1_rate_curve import build_di1_curve
    from kalman_price_mapper import build_ind_bova11_mapper_intraday

    mt5_conn = MT5Connector()
    build_di1_curve(mt5_conn)

    try:
        _, win_symbol = mt5_conn.get_symbol_futures("*WIN*")
    except:
        win_symbol = "WIN$N"

    mapper = None
    for ind_sym in ["WIN$N", win_symbol]:
        try:
            mapper = build_ind_bova11_mapper_intraday(
                mt5_conn, ind_symbol=ind_sym, bova11_symbol="BOVA11", max_days=10)
            break
        except:
            pass
    if mapper is None:
        print("[!] Could not build mapper"); return

    # Get cached dates
    cache_dir = os.path.join(SCRIPT_DIR, ".b3_cache")
    cached_dates = []
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith("COTAHIST_D") and f.endswith(".ZIP"):
            try:
                ds = f.replace("COTAHIST_D", "").replace(".ZIP", "")
                cached_dates.append(datetime.strptime(ds, "%d%m%Y"))
            except ValueError:
                continue

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    last_week = sorted([d for d in cached_dates if d < today and d.weekday() < 5])[-5:]
    if not last_week:
        print("[!] No cached data"); return

    # Pre-fetch all day data
    print(f"Preparing data for {len(last_week)} days...")
    day_data = []
    for day_dt in last_week:
        day_str = day_dt.strftime("%Y-%m-%d")
        bova_daily = mt5_conn.get_data("BOVA11", mt5.TIMEFRAME_D1, 10, 0)
        if bova_daily is None or bova_daily.empty: continue
        bova_daily['date'] = bova_daily['time'].dt.date
        day_bar = bova_daily[bova_daily['date'] <= day_dt.date()]
        if day_bar.empty: continue
        spot_bova = float(day_bar.iloc[-1]['open'])

        cw, pw, gf, sup, res = _compute_gex_walls_for_day(day_str, spot_bova)
        if cw is None or not np.isfinite(cw): continue

        bars_per_day = 90; total_bars = bars_per_day * 10
        win_intra = mt5_conn.get_data(win_symbol, mt5.TIMEFRAME_M5, total_bars, 0)
        if win_intra is None or win_intra.empty:
            win_intra = mt5_conn.get_data("WIN$N", mt5.TIMEFRAME_M5, total_bars, 0)
        if win_intra is None or win_intra.empty: continue

        win_intra['date'] = win_intra['time'].dt.date
        day_bars = win_intra[win_intra['date'] == day_dt.date()].copy().reset_index(drop=True)
        if day_bars.empty: continue

        bova_intra = mt5_conn.get_data("BOVA11", mt5.TIMEFRAME_M5, total_bars, 0)
        bova_day = pd.DataFrame()
        if bova_intra is not None and not bova_intra.empty:
            bova_intra['date'] = bova_intra['time'].dt.date
            bova_day = bova_intra[bova_intra['date'] == day_dt.date()].copy()

        day_data.append({
            'day_str': day_str, 'day_bars': day_bars, 'bova_day': bova_day,
            'cw': cw, 'pw': pw, 'gf': gf, 'sup': sup, 'res': res,
        })

    print(f"Loaded {len(day_data)} trading days.\n")

    # ── Also analyze entry times of all trades with wide-open window ──
    print("="*80)
    print("  STEP 1: Entry time analysis (window 09:00 - 17:30)")
    print("="*80)
    # Patch both the constants module AND the backtest module's local copies
    orig_start = bt.GEX_TRADE_WINDOW_START
    orig_end   = bt.GEX_TRADE_WINDOW_END
    bt.GEX_TRADE_WINDOW_START = "09:00"
    bt.GEX_TRADE_WINDOW_END   = "17:30"

    all_trades_wide = []
    for dd in day_data:
        # We need to override the constants used in simulate_day
        # simulate_day reads from constants directly
        trades = simulate_day(
            day_label=dd['day_str'], win_bars=dd['day_bars'], bova_bars=dd['bova_day'],
            call_wall_bova=dd['cw'], put_wall_bova=dd['pw'],
            gamma_flip_bova=dd['gf'], mapper=mapper,
            support_zones=dd['sup'], resist_zones=dd['res'],
        )
        for t in trades:
            t['entry_hour'] = t['entry_time'].strftime("%H:%M") if hasattr(t['entry_time'], 'strftime') else "?"
        all_trades_wide.extend(trades)

    if all_trades_wide:
        df = pd.DataFrame(all_trades_wide)
        print(f"\n  {'Day':<12} {'Side':<5} {'Entry Hour':<12} {'Exit Type':<6} {'P&L R$':>10}")
        print(f"  {'-'*50}")
        for _, t in df.iterrows():
            p = '+' if t['pnl_r'] >= 0 else ''
            print(f"  {t['day']:<12} {t['side']:<5} {t['entry_hour']:<12} {t['exit_type']:<6} {p}R${t['pnl_r']:>9,.2f}")

        # Group by hour bucket
        df['hour'] = pd.to_datetime(df['entry_hour'], format='%H:%M').dt.hour
        print(f"\n  {'ENTRY HOUR ANALYSIS':^60}")
        print(f"  {'-'*60}")
        print(f"  {'Hour':<8} {'Trades':>6} {'Wins':>6} {'Win%':>6} {'Total P&L':>12} {'Avg P&L':>10}")
        print(f"  {'-'*60}")
        for h in sorted(df['hour'].unique()):
            hdf = df[df['hour'] == h]
            n = len(hdf)
            w = (hdf['pnl_r'] > 0).sum()
            wr = w/n*100 if n > 0 else 0
            tp = hdf['pnl_r'].sum()
            ap = hdf['pnl_r'].mean()
            bar = '+' if tp >= 0 else '-'
            print(f"  {h:02d}:00    {n:>6} {w:>6} {wr:>5.0f}% R${tp:>+10,.2f} R${ap:>+8,.2f}")
    else:
        print("  No trades with wide window.")

    # ── Sweep different windows ──
    print(f"\n{'='*80}")
    print(f"  STEP 2: Time window sweep")
    print(f"{'='*80}")

    starts = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "12:00"]
    ends   = ["15:00", "15:30", "16:00", "16:30", "17:00", "17:30"]

    results = []
    for s in starts:
        for e in ends:
            if s >= e: continue
            bt.GEX_TRADE_WINDOW_START = s
            bt.GEX_TRADE_WINDOW_END   = e
            window_trades = []
            for dd in day_data:
                trades = simulate_day(
                    day_label=dd['day_str'], win_bars=dd['day_bars'], bova_bars=dd['bova_day'],
                    call_wall_bova=dd['cw'], put_wall_bova=dd['pw'],
                    gamma_flip_bova=dd['gf'], mapper=mapper,
                    support_zones=dd['sup'], resist_zones=dd['res'],
                )
                window_trades.extend(trades)
            n = len(window_trades)
            pnl = sum(t['pnl_r'] for t in window_trades)
            wins = sum(1 for t in window_trades if t['pnl_r'] > 0)
            wr = wins/n*100 if n > 0 else 0
            results.append({'start': s, 'end': e, 'trades': n, 'wins': wins,
                            'win_rate': wr, 'pnl': pnl})

    # Restore
    bt.GEX_TRADE_WINDOW_START = orig_start
    bt.GEX_TRADE_WINDOW_END   = orig_end

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values('pnl', ascending=False)

    print(f"\n  {'Start':<8} {'End':<8} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'Total P&L':>12}")
    print(f"  {'-'*52}")
    for _, r in rdf.head(20).iterrows():
        p = '+' if r['pnl'] >= 0 else ''
        print(f"  {r['start']:<8} {r['end']:<8} {r['trades']:>6} {r['wins']:>5} {r['win_rate']:>5.0f}% {p}R${r['pnl']:>10,.2f}")

    # Best window
    if not rdf.empty:
        best = rdf.iloc[0]
        print(f"\n  BEST WINDOW: {best['start']} - {best['end']} "
              f"({best['trades']} trades, {best['win_rate']:.0f}% WR, R${best['pnl']:+,.2f})")


if __name__ == "__main__":
    run_sweep()

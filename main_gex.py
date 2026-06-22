# -*- coding: utf-8 -*-
"""
Options GEX Analytics
---------------------
BOVA11 — B3 Brazilian options via COTAHIST + OI proxy

Performs:
- Global and range-based Put/Call Ratio
- IV skew (OTM puts vs OTM calls)
- Notional by strike (volume financeiro)
- Gamma Exposure (Customer/Dealer)
- Call/Put walls and Gamma Flip

Practical Usage — Intraday Trading
-----------------------------------
Best days to run:
  Mon/Tue → most reliable GEX levels (full gamma profile after Friday expiry).
  Wed     → mid-week check; levels still hold well.
  Thu/Fri → levels degrade as short-dated gamma dominates; use Friday GEX section.

Recommended intraday timeframe: 15-minute bars.
  - Dealer hedging rebalances are visible at this granularity.
  - Clean wall tests (call wall = resistance, put wall = support).
  - Drop to 5-min if spot is within ±0.5% of gamma flip (transition zone).

Session workflow:
  1. Pre-market (09:00 BRT): run script → note call wall, put wall, gamma flip.
  2. 10:00–11:30: first 6 bars — price discovery vs GEX levels.
  3. Wall touch on 15-min close → mean-reversion entry (positive gamma regime).
  4. Wall break on 15-min close → trend continuation (negative gamma regime).
  5. 14:00–16:00: strongest dealer hedging flow; 15-min signals most reliable.
"""
import numpy as np
import pandas as pd
import os
import sys
import asyncio

# Ensure parent dir is on sys.path for mt5_connector / get_b3_data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)

from constants import ASSET_SYMBOL, PLOT_GEX
from constants import GEX_SEND_ORDERS, GEX_MONITOR_ENABLED
from gex_utils import find_gamma_flip, compute_weekly_walls, generate_gex_trade_signals
from gex_plots import plot_gex_weekly
from b3_options_loader import load_b3_options_data

from flyagonal_strategy import build_flyagonal, format_flyagonal_snapshot, select_best_flyagonal
from strangle_strategy import build_strangle, format_strangle_snapshot
from mt5_connector import MT5Connector
from di1_rate_curve import build_di1_curve
from kalman_price_mapper import build_ind_bova11_mapper, build_ind_bova11_mapper_intraday
from gex_monitor import monitor_gex_entries
from gex_zones import (
    select_significant_zones,
    nearest_support_resistance,
    apply_proximity_offset,
    build_focus_expiry_snapshot,
    build_pin_candidates_snapshot,
    classify_sentiment_from_pcr,
    hedging_state as compute_hedging_state,
    format_gex_compact,
    format_oi,
    strength_label,
)
from gex_csv_export import export_gex_csv


async def analyze_options(spot: float, underlying: str = "PETR4", win_mapper=None,
                          win_symbol: str = "", mt5_conn=None):
    """
    Fetch options data from B3, compute Greeks via Black-Scholes, and analyze.
    Spot is passed as a parameter so the analysis aligns with current price.
    win_mapper: KalmanPriceMapper for converting BOVA11 levels to WIN current symbol (only for BOVA11).
    win_symbol: current WIN futures contract name (e.g. "WINM26").
    mt5_conn: MT5Connector instance for placing pending orders.
    """

    df = load_b3_options_data(underlying, spot)
    if df.empty:
        print(f"[X] No options data available for {underlying}")
        return


    df = df.dropna(subset=['Strike', 'IV', 'Gamma'])
    df = df[df['Strike'] > 0]

    calls = df[df['Tipo'].str.upper().str.contains('CALL')]
    puts  = df[df['Tipo'].str.upper().str.contains('PUT')]

    # ------------------------------------------------------------
    # PUT/CALL RATIO (global sentiment)
    # ------------------------------------------------------------
    total_calls = calls['Tit.'].sum()
    total_puts  = puts['Tit.'].sum()
    pcr_global = total_puts / total_calls if total_calls > 0 else np.nan

    print(f"\n===== STOCK OPTIONS -- Global PCR =====")
    print(f"Spot: {spot:.2f}")
    print(f"Total Calls: {total_calls:,.2f}")
    print(f"Total Puts : {total_puts:,.2f}")
    print(f"Put/Call Ratio: {pcr_global:.2f}")

    # ------------------------------------------------------------
    # IV SKEW — OTM puts vs OTM calls
    # ------------------------------------------------------------
    puts_otm  = puts[puts['Strike'] < spot]
    calls_otm = calls[calls['Strike'] > spot]
    iv_puts_otm  = puts_otm['IV'].mean() * 100
    iv_calls_otm = calls_otm['IV'].mean() * 100
    iv_skew = iv_puts_otm - iv_calls_otm

    print(f"\n===== Implied Volatility Skew =====")
    print(f"OTM Puts IV : {iv_puts_otm:.2f}%")
    print(f"OTM Calls IV: {iv_calls_otm:.2f}%")
    print(f"Skew (Puts - Calls): {iv_skew:.2f}%")

    # ------------------------------------------------------------
    # PCR BY STRIKE RANGE
    # ------------------------------------------------------------
    bins = [
        (0, 0.95*spot),
        (0.95*spot, 0.99*spot),
        (0.99*spot, 1.01*spot),
        (1.01*spot, 1.05*spot),
        (1.05*spot, np.inf),
    ]
    rows = []
    for (low, high) in bins:
        label = f"{low:.2f}-{high if np.isfinite(high) else 'Inf'}"
        c = calls[(calls['Strike']>=low)&(calls['Strike']<high)]['Tit.'].sum()
        p = puts[(puts['Strike']>=low)&(puts['Strike']<high)]['Tit.'].sum()
        pcr = p/c if c>0 else np.nan
        rows.append((label, c, p, pcr))
    df_pcr = pd.DataFrame(rows, columns=['Strike Range','Calls','Puts','PCR'])
    print(f"\n===== PCR by Strike Range =====")
    print(df_pcr)

    # ------------------------------------------------------------
    # GAMMA EXPOSURE (Customer)  —  Dollar Gamma = bs_gamma × S² × OI
    # ------------------------------------------------------------
    df['GEX_customer'] = df['Gamma'] * (spot ** 2) * df['Tit.']
    df['GEX_customer'] = df['GEX_customer'] * np.where(df['Tipo'].str.upper().str.contains('CALL'), 1, -1)

    gex_by_strike = df.groupby('Strike', as_index=False).agg(
        GEX_customer=('GEX_customer','sum')
    ).sort_values('Strike')

    # ============================================================
    # GEX FOR CURRENT WEEK & NEXT WEEK (Weekly Gamma Walls)
    # ============================================================
    weekly_results = compute_weekly_walls(df, spot)

    print(f"\n{'='*75}")
    print(f"WEEKLY GAMMA WALLS -- Current Week & Next Week")
    print(f"{'='*75}")

    avail_exp = sorted(pd.to_datetime(df['Expiration']).dt.date.unique())
    for wk in weekly_results:
        if wk['gex_by_strike'].empty:
            print(f"\n  {wk['label']}: No options expiring on {wk['friday_str']}")
            print(f"  Available expirations: {[str(d) for d in avail_exp[:10]]}")
            continue

        n_calls = len(wk['calls'])
        n_puts  = len(wk['puts'])
        print(f"\n  {wk['label']}: {wk['friday_str']} ({wk['dte']} BD)")
        print(f"    Contracts: {n_calls} calls, {n_puts} puts")
        print(f"    Total GEX: {wk['total_gex']/1e6:>10.2f}M")
        print(f"    Peak GEX strike: {wk['peak_gex_strike']:.2f}")
        if np.isfinite(wk['gamma_flip']):
            print(f"    Gamma Flip: {wk['gamma_flip']:.2f}")
        if np.isfinite(wk['call_wall']):
            print(f"    Call Wall:  {wk['call_wall']:.2f}")
        if np.isfinite(wk['put_wall']):
            print(f"    Put Wall:   {wk['put_wall']:.2f}")

    # Combined Call/Put Walls — current + next week expirations only
    combined_wk_dfs = [wk['gex_by_strike'] for wk in weekly_results if not wk['gex_by_strike'].empty]
    if combined_wk_dfs:
        combined_gex = pd.concat(combined_wk_dfs).groupby('Strike', as_index=False).agg(
            GEX_customer=('GEX_customer', 'sum')
        ).sort_values('Strike')
    else:
        combined_gex = gex_by_strike

    combined_fridays = [wk['friday_date'] for wk in weekly_results]
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    wk_mask = df['Expiration'].dt.date.isin([f.date() for f in combined_fridays])
    df_2wk = df[wk_mask] if wk_mask.any() else df

    gex_calls = df_2wk[df_2wk['Tipo'].str.upper().str.contains('CALL')]
    gex_puts  = df_2wk[df_2wk['Tipo'].str.upper().str.contains('PUT')]

    call_gex_by_strike = gex_calls.groupby('Strike')['GEX_customer'].sum()
    call_gex_above = call_gex_by_strike[call_gex_by_strike.index >= spot]
    call_wall = call_gex_above.idxmax() if not call_gex_above.empty else np.nan

    put_gex_by_strike = gex_puts.groupby('Strike')['GEX_customer'].sum()
    put_gex_below = put_gex_by_strike[put_gex_by_strike.index <= spot]
    put_wall = put_gex_below.abs().idxmax() if not put_gex_below.empty else np.nan

    gamma_flip = find_gamma_flip(df_2wk, spot)

    wk_labels = " + ".join(wk['friday_str'] for wk in weekly_results)
    print(f"\n===== Combined Walls (Current + Next Week: {wk_labels}) =====")
    print(f"Call Wall: {call_wall:.2f}")
    print(f"Put  Wall: {put_wall:.2f}")
    print(f"Gamma Flip (approx): {gamma_flip:.2f}")

    # Extended Market Structure Metrics
    print("\n" + "="*75)
    print("EXTENDED MARKET STRUCTURE METRICS -- STOCK TRACE-Lite View")
    print("="*75)

    print(f"Put/Call Ratio (OI):  {pcr_global:>6.2f}")
    if 0.9 <= pcr_global <= 1.1:
        sentiment = "Neutral"
    elif pcr_global > 1.1:
        sentiment = "Bearish - put demand dominates"
    else:
        sentiment = "Bullish - call demand dominates"
    print(f"Sentiment:            {sentiment}")

    print("\nVolatility Skew:")
    print(f"IV (OTM Puts):   {iv_puts_otm:>6.2f}%")
    print(f"IV (OTM Calls):  {iv_calls_otm:>6.2f}%")
    print(f"Skew (Puts-Calls): {iv_skew:>6.2f}%")

    if iv_skew > 10:
        print("Interpretation:  Elevated skew -- investors hedging downside risk.")
    elif iv_skew < 0:
        print("Interpretation:  Inverted skew -- speculative upside bias.")
    else:
        print("Interpretation:  Balanced implied vol surface.")

    print("\nGamma Flip Analysis:")
    print(f"Gamma Flip (approx): {gamma_flip:>8.2f}")
    print(f"Spot:                 {spot:>8.2f}")

    if np.isfinite(gamma_flip):
        diff = spot - gamma_flip
        pct  = diff / gamma_flip * 100
        side = "above" if diff > 0 else "below"
        print(f"Spot is {abs(pct):.2f}% {side} the flip.")
        if diff > 0:
            print("-> Dealers long gamma: market mechanically dampened.")
        else:
            print("-> Dealers short gamma: market mechanically amplified.")

    # Market regime classification (positive gamma: spot > gamma_flip)
    if np.isfinite(gamma_flip):
        if spot >= gamma_flip * 1.05:
            regime = "POSITIVE GAMMA (Low Volatility)"
            rationale = "Dealers long gamma, hedging dampens volatility (mean-reverting)."
            strategy = "Range trading, mean-reversion, sell call wall, buy put wall."
        elif spot <= gamma_flip * 0.95:
            regime = "NEGATIVE GAMMA (High Volatility)"
            rationale = "Dealers short gamma, hedging amplifies volatility (trending)."
            strategy = "Trend following, breakout trades, buy above gamma flip."
        else:
            regime = "TRANSITION ZONE"
            rationale = "Market near flip - unstable hedging behavior."
            strategy = "Reduce size, use 5-min confirmation, neutral setups."
    else:
        regime, rationale, strategy = "UNKNOWN", "Gamma Flip not found", "N/A"

    print("\nMarket Regime:")
    print(f"Detected:     {regime}")
    print(f"Rationale:    {rationale}")
    print(f"Recommended:  {strategy}")

    # Significant GEX zones — dashboard-style support / resistance summary
    zone_source = combined_gex.copy() if not combined_gex.empty else gex_by_strike.copy()
    resist_zones, support_zones = select_significant_zones(zone_source, spot, top_n=3, zone_pct=0.04)
    focus_snapshot = build_focus_expiry_snapshot(df, spot, top_n=3, zone_pct=0.04)
    pin_snapshot = build_pin_candidates_snapshot(df, spot, top_n=5, pct_range=0.05)
    if not focus_snapshot['resist_zones'].empty or not focus_snapshot['support_zones'].empty:
        resist_zones = focus_snapshot['resist_zones']
        support_zones = focus_snapshot['support_zones']

    flip_dist_pct = ((spot - gamma_flip) / gamma_flip * 100.0) if np.isfinite(gamma_flip) and gamma_flip != 0 else np.nan
    sentiment_pt = classify_sentiment_from_pcr(pcr_global)
    hedging_state_label = compute_hedging_state(spot, gamma_flip)

    print("\n" + "="*75)
    print(f"GEX SNAPSHOT SUMMARY -- {underlying}")
    print("="*75)
    _wlabel = win_symbol if win_symbol else "WIN"
    if win_mapper is not None:
        cw_win = f" ({_wlabel} {win_mapper.bova11_to_ind(call_wall):,.0f})" if np.isfinite(call_wall) else ""
        pw_win = f" ({_wlabel} {win_mapper.bova11_to_ind(put_wall):,.0f})" if np.isfinite(put_wall) else ""
        flip_win = f" ({_wlabel} {win_mapper.bova11_to_ind(gamma_flip):,.0f})" if np.isfinite(gamma_flip) else ""
        spot_win = f" ({_wlabel} {win_mapper.bova11_to_ind(spot):,.0f})"
    else:
        cw_win = pw_win = flip_win = spot_win = ""
    print(f"WALLS (C/P): {(f'{call_wall:.2f}' if np.isfinite(call_wall) else 'N/A')}{cw_win} / {(f'{put_wall:.2f}' if np.isfinite(put_wall) else 'N/A')}{pw_win}")
    print(f"GAMMA FLIP : {gamma_flip:.2f}{flip_win}" if np.isfinite(gamma_flip) else "GAMMA FLIP : N/A")
    print(f"PCR (OI)   : {pcr_global:.2f}")
    print(f"SPOT       : {spot:.2f}{spot_win}")
    print(f"SENTIMENTO : {sentiment_pt}")
    print(f"IV SKEW    : {iv_skew:.2f}%")
    print(f"REGIME     : {regime}")
    print(f"FLIP DIST. : {flip_dist_pct:+.2f}%" if np.isfinite(flip_dist_pct) else "FLIP DIST. : N/A")
    print(f"HEDGING    : {hedging_state_label}")
    if focus_snapshot['expiry_label'] != 'N/A':
        dte_label = f"{focus_snapshot['dte']} DTE" if np.isfinite(focus_snapshot['dte']) else "N/A"
        print(f"FOCUS EXP. : {focus_snapshot['expiry_label']} ({dte_label})")

    if not pin_snapshot['pin_candidates'].empty:
        pin_dte_label = f"{pin_snapshot['dte']} DTE" if np.isfinite(pin_snapshot['dte']) else "N/A"
        print("\nPIN CANDIDATES (+/-5% FROM SPOT) SNAPSHOT:")
        print(f"Expiry: {pin_snapshot['expiry_label']} ({pin_dte_label})")
        if win_mapper is not None:
            print(f"{'Strike':>10} {_wlabel:>10} {'Dealer GEX':>14} {'Calls OI':>12} {'Puts OI':>12}")
            print("-" * 66)
        else:
            print(f"{'Strike':>10} {'Dealer GEX':>14} {'Calls OI':>12} {'Puts OI':>12}")
            print("-" * 54)
        for _, row in pin_snapshot['pin_candidates'].iterrows():
            win_str = f"{win_mapper.bova11_to_ind(row['Strike']):>10,.0f} " if win_mapper is not None else ""
            print(
                f"{row['Strike']:>10.2f} "
                f"{win_str}"
                f"{format_gex_compact(row['dealer_gex']):>14} "
                f"{format_oi(row['call_oi']):>12} "
                f"{format_oi(row['put_oi']):>12}"
            )

    print("\nZONAS SIGNIFICATIVAS GEX:")
    if win_mapper is not None:
        print(f"{'ZONA':<14} {'STRIKE':>10} {_wlabel:>10} {'GEX':>14} {'STRENGTH':>10}")
        print("-" * 66)
    else:
        print(f"{'ZONA':<14} {'STRIKE':>10} {'GEX':>14} {'STRENGTH':>10}")
        print("-" * 54)

    if resist_zones.empty and support_zones.empty:
        print(f"{'N/A':<14} {'-':>10} {'-':>14} {'-':>10}")
    else:
        for _, row in resist_zones.iterrows():
            win_str = f" {win_mapper.bova11_to_ind(row['Strike']):>10,.0f}" if win_mapper is not None else ""
            print(f"{'RESISTENCIA':<14} {row['Strike']:>10.2f}{win_str} {format_gex_compact(row['GEX_customer']):>14} {strength_label(row['GEX_customer']):>10}")
        for _, row in support_zones.iterrows():
            win_str = f" {win_mapper.bova11_to_ind(row['Strike']):>10,.0f}" if win_mapper is not None else ""
            print(f"{'SUPORTE':<14} {row['Strike']:>10.2f}{win_str} {format_gex_compact(row['GEX_customer']):>14} {strength_label(row['GEX_customer']):>10}")

    print("\nTOP 3 RESISTANCE ZONES:")
    if resist_zones.empty:
        print("  N/A")
    else:
        for idx, (_, row) in enumerate(resist_zones.reset_index(drop=True).iterrows(), start=1):
            win_str = f" ({_wlabel} {win_mapper.bova11_to_ind(row['Strike']):,.0f})" if win_mapper is not None else ""
            print(f"  {idx}. Strike {row['Strike']:.2f}{win_str} | GEX {format_gex_compact(row['GEX_customer'])}")

    print("\nTOP 3 SUPPORT ZONES:")
    if support_zones.empty:
        print("  N/A")
    else:
        for idx, (_, row) in enumerate(support_zones.reset_index(drop=True).iterrows(), start=1):
            win_str = f" ({_wlabel} {win_mapper.bova11_to_ind(row['Strike']):,.0f})" if win_mapper is not None else ""
            print(f"  {idx}. Strike {row['Strike']:.2f}{win_str} | GEX {format_gex_compact(row['GEX_customer'])}")

    # Summary Snapshot
    def _fmt(label, bova_val):
        if not np.isfinite(bova_val):
            return f"  {label:<25s} N/A"
        if win_mapper is not None:
            win_val = win_mapper.bova11_to_ind(bova_val)
            return f"  {label:<25s} {bova_val:>10,.2f}   |  {_wlabel} {win_val:>10,.0f}"
        return f"  {label:<25s} {bova_val:>10,.2f}"

    header = "BOVA11" if win_mapper is not None else underlying
    win_hdr = f"  |  {_wlabel}" if win_mapper is not None else ""

    print(f"\nSummary Snapshot ({header}{win_hdr}):")
    if win_mapper is not None:
        win_spot = win_mapper.bova11_to_ind(spot)
        print(f"  {'Spot:':<25s} {spot:>10,.2f}   |  {_wlabel} {win_spot:>10,.0f}")
    else:
        print(f"  {'Spot:':<25s} {spot:>10,.2f}")

    print(f"\nWalls by Expiration Date:")
    for wk in weekly_results:
        if wk['gex_by_strike'].empty:
            print(f"  {wk['friday_str']}  -- No data")
            continue
        wk_cw = wk['call_wall']
        wk_pw = wk['put_wall']
        wk_flip = wk['gamma_flip']
        print(f"\n  {wk['friday_str']} ({wk['label']}, {wk['dte']} BD):")
        print(_fmt("Call Wall:", wk_cw))
        print(_fmt("Put Wall:", wk_pw))
        print(_fmt("Gamma Flip:", wk_flip))

    # Support & Resistance zones — average of weekly walls + gamma flip
    valid_cw = [wk['call_wall'] for wk in weekly_results
                if not wk['gex_by_strike'].empty and np.isfinite(wk['call_wall'])]
    valid_pw = [wk['put_wall'] for wk in weekly_results
                if not wk['gex_by_strike'].empty and np.isfinite(wk['put_wall'])]
    valid_gf = [wk['gamma_flip'] for wk in weekly_results
                if not wk['gex_by_strike'].empty and np.isfinite(wk['gamma_flip'])]
    avg_resistance = np.mean(valid_cw) if valid_cw else np.nan
    avg_support = np.mean(valid_pw) if valid_pw else np.nan
    avg_gamma_flip = np.mean(valid_gf) if valid_gf else np.nan

    print(f"\nKey Zones (avg of weekly walls):")
    print(_fmt("Resistance (Call Wall):", avg_resistance))
    print(_fmt("Support (Put Wall):", avg_support))
    print(_fmt("Gamma Flip:", avg_gamma_flip))

    print(f"\nMarket Regime: {regime}")
    print("="*75)

    # ----------------------------------------------------------------
    # GEX TRADE SIGNAL — regime + nearest S/R zone proximity
    # ----------------------------------------------------------------
    trade_signal = generate_gex_trade_signals(
        spot, gamma_flip, call_wall, put_wall,
        support_zones=support_zones if not support_zones.empty else None,
        resist_zones=resist_zones if not resist_zones.empty else None,
    )

    strength_bar = '#' * trade_signal['strength'] + '.' * (3 - trade_signal['strength'])

    print(f"\n{'='*75}")
    print(f"GEX TRADE SIGNAL -- {underlying}")
    print(f"{'='*75}")
    print(f"  SIGNAL   : {trade_signal['signal']}  [{strength_bar}]")
    print(f"  REGIME   : {trade_signal['regime']}")
    print(f"  STRENGTH : {trade_signal['strength']}/3")
    print(f"  REASON   : {trade_signal['reason']}")
    print(f"{'='*75}")

    # ----------------------------------------------------------------
    # GEX ENTRY LEVELS — closest support / resistance to spot + offset
    # ----------------------------------------------------------------
    entry_buy, entry_sell = nearest_support_resistance(
        spot, support_zones, resist_zones,
        put_wall=put_wall, call_wall=call_wall,
        pin_candidates=pin_snapshot.get('pin_candidates', pd.DataFrame()),
    )
    entry_buy = apply_proximity_offset(entry_buy, 'buy')
    entry_sell = apply_proximity_offset(entry_sell, 'sell')

    if win_mapper is not None:
        _eb = f" (WIN {win_mapper.bova11_to_ind(entry_buy):.0f})" if np.isfinite(entry_buy) else ""
        _es = f" (WIN {win_mapper.bova11_to_ind(entry_sell):.0f})" if np.isfinite(entry_sell) else ""
    else:
        _eb = _es = ""
    print(f"\n  ENTRY BUY  : {f'{entry_buy:.2f}' if np.isfinite(entry_buy) else 'N/A'}{_eb}")
    print(f"  ENTRY SELL : {f'{entry_sell:.2f}' if np.isfinite(entry_sell) else 'N/A'}{_es}")

    # ----------------------------------------------------------------
    # FLYAGONAL STRATEGY — Diagonal Butterfly from GEX levels
    # Build both call and put variants; best one shown at end of scan.
    # ----------------------------------------------------------------
    _fly_call = build_flyagonal(
        df, spot, weekly_results, pin_snapshot, regime,
        option_type='call',
    )
    _fly_put = build_flyagonal(
        df, spot, weekly_results, pin_snapshot, regime,
        option_type='put',
    )
    flyagonal = select_best_flyagonal([_fly_call, _fly_put])

    # ----------------------------------------------------------------
    # STRANGLE STRATEGY — OTM call + put using GEX walls as strikes
    # ----------------------------------------------------------------
    strangle = build_strangle(
        df, spot, weekly_results, call_wall, put_wall, regime,
    )

    # ----------------------------------------------------------------
    # Export CSV for MQL5 indicator → MQL5/Files/GEX_<underlying>.csv
    # ----------------------------------------------------------------
    export_gex_csv(
        underlying, spot, call_wall, put_wall, gamma_flip, regime,
        weekly_results, pin_snapshot, resist_zones, support_zones,
        win_mapper,
        trade_signal=trade_signal,
        flyagonal=flyagonal,
        strangle=strangle,
        win_symbol=win_symbol,
    )

    if PLOT_GEX:
        plot_gex_weekly(
            weekly_results, spot, underlying,
            pin_candidates=pin_snapshot['pin_candidates'] if not pin_snapshot['pin_candidates'].empty else None,
            resist_zones=resist_zones if not resist_zones.empty else None,
            support_zones=support_zones if not support_zones.empty else None,
            win_mapper=win_mapper,
            show_plots=True,
        )

    # ----------------------------------------------------------------
    # BEST FLYAGONAL SETUP — end-of-scan summary
    # ----------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"BEST FLYAGONAL SETUP -- END OF SCAN")
    print(f"{'='*75}")
    print(format_flyagonal_snapshot(flyagonal, win_mapper=win_mapper))

    # ----------------------------------------------------------------
    # STRANGLE SETUP — end-of-scan summary
    # ----------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"STRANGLE SETUP -- END OF SCAN")
    print(f"{'='*75}")
    print(format_strangle_snapshot(strangle, win_mapper=win_mapper))

    return {
        'call_wall': call_wall,
        'put_wall': put_wall,
        'gamma_flip': gamma_flip,
        'regime': regime,
        'trade_signal': trade_signal,
        'support_zones': support_zones,
        'resist_zones': resist_zones,
        'pin_candidates': pin_snapshot.get('pin_candidates', pd.DataFrame()),
        'strangle': strangle,
    }


async def main():
    print("[DEBUG] main() started")
    mt5_conn = MT5Connector()
    # --- Debug: Print current WIN and BOVA11 prices ---
    win_symbol, prev_win = mt5_conn.get_win_symbols()
    bova_info = mt5_conn.get_symbol_info("BOVA11")
    win_info = mt5_conn.get_symbol_info(win_symbol) if win_symbol else None
    bova_price = (bova_info.bid + bova_info.ask) / 2 if bova_info and bova_info.bid > 0 and bova_info.ask > 0 else None
    win_price = (win_info.bid + win_info.ask) / 2 if win_info and win_info.bid > 0 and win_info.ask > 0 else None
    if bova_info:
        print(f"[DEBUG] BOVA11 price: {bova_price} (bid={bova_info.bid}, ask={bova_info.ask})")
    else:
        print("[DEBUG] Could not fetch BOVA11 symbol info")
    if win_info:
        print(f"[DEBUG] WIN price: {win_price} (bid={win_info.bid}, ask={win_info.ask}) [{win_symbol}]")
    else:
        print(f"[DEBUG] Could not fetch WIN symbol info for {win_symbol}")

    # --- After mapper is built, print regression parameters and mapping ---
    # This will be after the win_mapper is built below

    # Build DI1 term-structure once (spline-interpolated per expiry)
    print("[DEBUG] DI1 curve build starting")
    build_di1_curve(mt5_conn)
    print("[DEBUG] DI1 curve build finished")

    # Build Kalman mapper WIN <-> BOVA11 on 15-min bars (best for intraday)
    win_mapper = None
    win_symbol = ""
    expiring_symbol = None
    print("[DEBUG] Starting WIN/BOVA11 Kalman mapper build")
    if "BOVA11" in ASSET_SYMBOL:
        try:
            (_exp_time, win_symbol), expiring_symbol = mt5_conn.get_symbol_futures(
                "*WIN*", include_expiring=True
            )
            print(f"[i] Current WIN futures contract: {win_symbol}")
            if expiring_symbol:
                print(f"[i] Expiring contract (data fallback): {expiring_symbol}")
        except Exception as e:
            print(f"[!] Could not resolve WIN symbols: {e}")
            win_symbol = ""
            expiring_symbol = None

        # Trading symbol first; expiring contract as historical-data fallback for mapper
        ind_symbols_to_try = [win_symbol] if win_symbol else []
        if expiring_symbol and expiring_symbol not in ind_symbols_to_try:
            ind_symbols_to_try.append(expiring_symbol)



        for ind_sym in ind_symbols_to_try:
            print(f"[DEBUG] Trying to build mapper for symbol: {ind_sym}")
            try:
                win_mapper = build_ind_bova11_mapper_intraday(
                    mt5_conn, ind_symbol=win_symbol, bova11_symbol="BOVA11"
                )
                print(f"[i] Intraday mapper built using {win_symbol}")
            except Exception as e:
                print(f"[!] Intraday mapper failed for {win_symbol}: {e}")
                try:
                    win_mapper = build_ind_bova11_mapper(mt5_conn, ind_symbol=win_symbol, bova11_symbol="BOVA11")
                    print(f"[i] Daily mapper built using {win_symbol}")
                except Exception as e2:
                    print(f"[!] Daily mapper also failed for {win_symbol}: {e2}")

        # Debug: Print Kalman regression parameters and sample conversion
        if win_mapper is not None:
            print(f"[DEBUG] KalmanPriceMapper alpha: {win_mapper.alpha}")
            print(f"[DEBUG] KalmanPriceMapper beta: {win_mapper.beta}")
            print(f"[DEBUG] KalmanPriceMapper exp(alpha): {np.exp(win_mapper.alpha)} (should be close to 1000)")
            # Try a sample conversion with the actual BOVA11 price
            if bova_price:
                win_est = win_mapper.bova11_to_ind(bova_price)
                print(f"[DEBUG] WIN estimate from BOVA11={bova_price}: {win_est}")
                if win_price:
                    print(f"[DEBUG] Actual WIN price: {win_price}")
                    diff = abs(win_est - win_price)
                    print(f"[DEBUG] Difference (mapped - actual): {diff}")
                else:
                    print("[DEBUG] Actual WIN price not available for comparison.")
            else:
                print("[DEBUG] BOVA11 price not available for mapping.")
            # Warn if regression is out of expected range
            if not (0.7 <= win_mapper.beta <= 1.3):
                print(f"[WARNING] Kalman regression beta out of expected range: {win_mapper.beta}")
            if not (900 <= np.exp(win_mapper.alpha) <= 1100):
                print(f"[WARNING] Kalman regression exp(alpha) out of expected range: {np.exp(win_mapper.alpha)}")

        if not ind_symbols_to_try:
            print("[!] No WIN contract resolved -- no mapper available")
        print("[DEBUG] Mapper build step finished")

    print("[DEBUG] Checking asset scope and monitor flags")
    _assets_upper = {str(a).upper() for a in ASSET_SYMBOL}
    _is_bova11_scope = "BOVA11" in _assets_upper
    _is_win_symbol = bool(win_symbol) and str(win_symbol).upper().startswith("WIN")
    _prioritize_live_monitor = (
        GEX_MONITOR_ENABLED and GEX_SEND_ORDERS and _is_bova11_scope and _is_win_symbol
    )

    print("[DEBUG] Preparing analysis asset list")
    analysis_assets = list(ASSET_SYMBOL)
    if _prioritize_live_monitor and "BOVA11" in analysis_assets:
        analysis_assets = ["BOVA11"] + [asset for asset in analysis_assets if asset != "BOVA11"]

    bova11_gex = None
    remaining_assets = []
    print("[DEBUG] Starting asset analysis loop")
    for asset in analysis_assets:
        print(f"\n{'#'*80}\nAnalyzing {asset}...\n{'#'*80}")
        symbol_info = mt5_conn.get_symbol_info(asset)
        if symbol_info is None:
            print(f"[X] Could not get symbol info for {asset} -- skipping.")
            continue
        spot_price = (symbol_info.bid + symbol_info.ask) / 2
        if spot_price <= 0:
            # Fallback: use last traded price (common at market open when bid/ask are 0)
            last_price = float(getattr(symbol_info, 'last', 0.0) or 0.0)
            if last_price > 0:
                spot_price = last_price
                print(f"[!] {asset}: bid/ask are 0 — using last traded price {spot_price:.2f}")
            else:
                print(f"[X] Spot price for {asset} is {spot_price:.2f} (bid={symbol_info.bid}, ask={symbol_info.ask}, last={last_price:.2f}) -- skipping.")
                continue
        print(f"Analyzing options data for {asset} with spot price {spot_price:.2f}...")
        mapper_for_asset = win_mapper if asset == "BOVA11" else None
        win_sym_for_asset = win_symbol if asset == "BOVA11" else ""
        print(f"[DEBUG] Calling analyze_options for {asset} with spot {spot_price}")
        try:
            result = await analyze_options(
                spot_price,
                asset,
                win_mapper=mapper_for_asset,
                win_symbol=win_sym_for_asset,
                mt5_conn=mt5_conn,
            )
        except Exception as _exc:
            print(f"[X] analyze_options raised an exception for {asset}: {_exc}")
            result = None
        print(f"[DEBUG] analyze_options finished for {asset}, result={'dict' if result is not None else 'None'}")
        if asset == "BOVA11" and result is not None:
            bova11_gex = result
            if _prioritize_live_monitor:
                remaining_assets = [name for name in analysis_assets if name != "BOVA11"]
                if remaining_assets:
                    print(f"[i] Deferring additional asset scans until after monitor session: {', '.join(remaining_assets)}")
                break

    # --- Start real-time GEX monitor only for BOVA11 -> WIN regression flow ---
    if (GEX_MONITOR_ENABLED and GEX_SEND_ORDERS
            and _is_bova11_scope
            and _is_win_symbol
            and bova11_gex is not None
            and win_mapper is not None and win_symbol):
        await monitor_gex_entries(
            mt5_conn, win_symbol, win_mapper,
            call_wall=bova11_gex['call_wall'],
            put_wall=bova11_gex['put_wall'],
            gamma_flip=bova11_gex['gamma_flip'],
            support_zones=bova11_gex.get('support_zones'),
            resist_zones=bova11_gex.get('resist_zones'),
            pin_candidates=bova11_gex.get('pin_candidates'),
        )
    elif GEX_MONITOR_ENABLED:
        reasons = []
        if not GEX_SEND_ORDERS:
            reasons.append("GEX_SEND_ORDERS is False")
        if not _is_bova11_scope:
            reasons.append("BOVA11 not in ASSET_SYMBOL (monitor restricted to BOVA11->WIN)")
        if not _is_win_symbol:
            reasons.append("WIN symbol not resolved/invalid (monitor restricted to WIN contracts)")
        if bova11_gex is None:
            if _is_bova11_scope:
                reasons.append("BOVA11 analysis returned None (spot price was 0, or B3 data unavailable, or exception in analyze_options)")
            else:
                reasons.append("BOVA11 not in ASSET_SYMBOL")
        if win_mapper is None:
            reasons.append("WIN-BOVA11 mapper unavailable")
        if not win_symbol:
            reasons.append("WIN symbol not resolved")
        print(f"\n[GEX Monitor] Cannot start: {'; '.join(reasons)}")


if __name__ == "__main__":
    print("[DEBUG] __main__ entry reached, running main()...")
    asyncio.run(main())

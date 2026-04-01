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
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from constants import ASSET_SYMBOL
from gex_utils import find_gamma_flip, compute_weekly_walls
from gex_plots import plot_notional_by_strike, plot_gex_all_expiry, plot_gex_weekly
from b3_options_loader import load_b3_options_data
from mt5_connector import MT5Connector
from di1_rate_curve import build_di1_curve
from kalman_price_mapper import build_ind_bova11_mapper, build_ind_bova11_mapper_intraday


async def analyze_options(spot: float, underlying: str = "PETR4", show_plots: bool = False, win_mapper=None):
       """
       Fetch options data from B3, compute Greeks via Black-Scholes, and analyze.
       Spot is passed as a parameter so the analysis aligns with current price.
       If show_plots is False, all matplotlib charts are suppressed.
       win_mapper: KalmanPriceMapper for converting BOVA11 levels to WIN$N (only for BOVA11).
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
           (0, 0.95*spot),          # Deep OTM puts
           (0.95*spot, 0.99*spot),  # Near OTM puts
           (0.99*spot, 1.01*spot),  # ATM range
           (1.01*spot, 1.05*spot),  # Near OTM calls
           (1.05*spot, np.inf),     # Far OTM calls
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
       # NOTIONAL (volume financeiro por strike)
       # ------------------------------------------------------------
       vol_by_strike = df.groupby(['Strike','Tipo'])['VolFin'].sum().unstack(fill_value=0)
       plot_notional_by_strike(vol_by_strike, spot, underlying, show_plots)

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

       plot_gex_weekly(weekly_results, spot, underlying, show_plots)

       # Combined Call/Put Walls — current + next week expirations only
       combined_wk_dfs = [wk['gex_by_strike'] for wk in weekly_results if not wk['gex_by_strike'].empty]
       if combined_wk_dfs:
           combined_gex = pd.concat(combined_wk_dfs).groupby('Strike', as_index=False).agg(
               GEX_customer=('GEX_customer', 'sum')
           ).sort_values('Strike')
       else:
           combined_gex = gex_by_strike  # fallback to all expiries

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
       put_wall  = put_gex_below.abs().idxmax() if not put_gex_below.empty else np.nan

       # Gamma Flip — scan-based: re-evaluate gamma at each test price
       gamma_flip = find_gamma_flip(df_2wk, spot)

       wk_labels = " + ".join(wk['friday_str'] for wk in weekly_results)
       print(f"\n===== Combined Walls (Current + Next Week: {wk_labels}) =====")
       print(f"Call Wall: {call_wall:.2f}")
       print(f"Put  Wall: {put_wall:.2f}")
       print(f"Gamma Flip (approx): {gamma_flip:.2f}")

       plot_gex_all_expiry(combined_gex, spot, underlying, gamma_flip,
                          call_wall, put_wall, show_plots)
   
       # Extended Market Structure Metrics
       print("\n" + "="*75)
       print("EXTENDED MARKET STRUCTURE METRICS -- STOCK TRACE-Lite View")
       print("="*75)
   
       print(f"Put/Call Ratio (OI):  {pcr_global:>6.2f}")
       if 0.9 <= pcr_global <= 1.1:
           sentiment = "Neutral"
       elif pcr_global > 1.1:
           sentiment = "Bearish — put demand dominates"
       else:
           sentiment = "Bullish — call demand dominates"
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
               rationale = "Market near flip — unstable hedging behavior."
               strategy = "Reduce size, use 5-min confirmation, neutral setups."
       else:
           regime, rationale, strategy = "UNKNOWN", "Gamma Flip not found", "N/A"
   
       print("\nMarket Regime:")
       print(f"Detected:     {regime}")
       print(f"Rationale:    {rationale}")
       print(f"Recommended:  {strategy}")
   
       # Significant GEX zones
       print("\nSignificant GEX Zones:")
       gex_sorted = gex_by_strike.sort_values("GEX_customer", ascending=False)
       resist = gex_sorted.head(4)
       supports = gex_sorted.tail(4)
   
       print("Support Zones (dealers short gamma on puts -> hedge-buying cushion):")
       for _, r in supports.iterrows():
           gex_mil = r["GEX_customer"] / 1e6
           strength = "Strong" if abs(gex_mil) > 200 else "Moderate" if abs(gex_mil) > 100 else "Weak"
           print(f"  Strike {r['Strike']:>8.2f} | {gex_mil:>7.2f}M | {strength}")
   
       print("\nResistance Zones (dealers long gamma -> counter-trend selling caps rallies):")
       for _, r in resist.iterrows():
           gex_mil = r["GEX_customer"] / 1e6
           strength = "Strong" if abs(gex_mil) > 200 else "Moderate" if abs(gex_mil) > 100 else "Weak"
           print(f"  Strike {r['Strike']:>8.2f} | +{gex_mil:>7.2f}M | {strength}")
   
       # Summary Snapshot
       # Helper to format BOVA11 + WIN$N side by side
       def _fmt(label, bova_val):
           if not np.isfinite(bova_val):
               return f"  {label:<25s} N/A"
           if win_mapper is not None:
               win_val = win_mapper.bova11_to_ind(bova_val)
               return f"  {label:<25s} {bova_val:>10,.2f}   |  WIN$N {win_val:>10,.0f}"
           return f"  {label:<25s} {bova_val:>10,.2f}"

       header = "BOVA11" if win_mapper is not None else underlying
       win_hdr = "  |  WIN$N" if win_mapper is not None else ""

       print(f"\nSummary Snapshot ({header}{win_hdr}):")
       if win_mapper is not None:
           win_spot = win_mapper.bova11_to_ind(spot)
           print(f"  {'Spot:':<25s} {spot:>10,.2f}   |  WIN$N {win_spot:>10,.0f}")
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

async def main():
    mt5_conn = MT5Connector()

    # Build DI1 term-structure once (spline-interpolated per expiry)
    build_di1_curve(mt5_conn)

    # Build Kalman mapper WIN$N <-> BOVA11 on 15-min bars (best for intraday)
    win_mapper = None
    if "BOVA11" in ASSET_SYMBOL:
        try:
            win_mapper = build_ind_bova11_mapper_intraday(
                mt5_conn, ind_symbol="WIN$N", bova11_symbol="BOVA11"
            )
        except Exception as e:
            print(f"[!] Could not build WIN-BOVA11 intraday mapper: {e}")
            # Fallback to daily mapper
            try:
                win_mapper = build_ind_bova11_mapper(mt5_conn, ind_symbol="WIN$N", bova11_symbol="BOVA11")
                print("[!] Falling back to daily mapper")
            except Exception as e2:
                print(f"[!] Daily mapper also failed: {e2}")

    for asset in ASSET_SYMBOL:
        print(f"\n{'#'*80}\nAnalyzing {asset}...\n{'#'*80}")
        symbol_info = mt5_conn.get_symbol_info(asset)
        if symbol_info is None:
            print(f"[X] Could not get symbol info for {asset} -- skipping.")
            continue
        spot_price = (symbol_info.bid + symbol_info.ask) / 2
        if spot_price <= 0:
            print(f"[X] Spot price for {asset} is {spot_price:.2f} (bid={symbol_info.bid}, ask={symbol_info.ask}) -- skipping.")
            continue
        print(f"Analyzing options data for {asset} with spot price {spot_price:.2f}...")
        mapper_for_asset = win_mapper if asset == "BOVA11" else None
        await analyze_options(spot_price, asset, show_plots=True, win_mapper=mapper_for_asset)

asyncio.run(main())
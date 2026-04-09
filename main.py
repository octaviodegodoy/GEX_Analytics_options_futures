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
from constants import GEX_SEND_ORDERS, GEX_ORDER_VOLUME, GEX_ORDER_DEVIATION, GEX_MIN_SIGNAL_STRENGTH
from constants import GEX_MONITOR_INTERVAL, GEX_MONITOR_ENABLED, GEX_MAGIC_NUMBER
from gex_utils import find_gamma_flip, compute_weekly_walls, generate_gex_trade_signals
from gex_plots import plot_notional_by_strike, plot_gex_all_expiry, plot_gex_weekly
from b3_options_loader import load_b3_options_data
from flyagonal_strategy import build_flyagonal, format_flyagonal_snapshot
from mt5_connector import MT5Connector
from di1_rate_curve import build_di1_curve
from kalman_price_mapper import build_ind_bova11_mapper, build_ind_bova11_mapper_intraday


def _classify_sentiment_from_pcr(pcr_global):
       """Map PCR to a short sentiment label used in the console snapshot."""
       if pcr_global is None or not np.isfinite(pcr_global):
           return "N/A"
       if pcr_global < 0.90:
           return "ALTISTA"
       if pcr_global > 1.10:
           return "BAIXISTA"
       return "NEUTRO"


def _hedging_state(spot, gamma_flip):
       """Return a simple hedging state label similar to the dashboard card."""
       if gamma_flip is None or not np.isfinite(gamma_flip) or gamma_flip == 0:
           return "N/A"
       dist_pct = abs((spot - gamma_flip) / gamma_flip) * 100.0
       if dist_pct <= 0.50:
           return "DAMPED"
       return "DAMPED" if spot >= gamma_flip else "AMPLIFIED"


def _format_gex_compact(value):
       """Human-readable GEX formatter (k / M / B) for console summaries."""
       if value is None or not np.isfinite(value):
           return "N/A"
       value = float(value)
       abs_value = abs(value)
       sign = "-" if value < 0 else ""

       if abs_value >= 1e9:
           return f"{sign}{abs_value / 1e9:.1f}B"
       if abs_value >= 1e6:
           return f"{sign}{abs_value / 1e6:.1f}M"
       if abs_value >= 1e3:
           return f"{sign}{abs_value / 1e3:.1f}k"
       return f"{value:.1f}"


def _select_significant_zones(gex_frame, spot, top_n=3, zone_pct=0.04):
       """
       Pick the strongest nearby resistance/support strikes around spot.

       - Resistance: positive GEX strikes at/above spot
       - Support:    negative GEX strikes at/below spot
       """
       if gex_frame is None or gex_frame.empty:
           return pd.DataFrame(), pd.DataFrame()

       working = gex_frame.copy()
       lo = spot * (1.0 - zone_pct)
       hi = spot * (1.0 + zone_pct)
       window = working[(working["Strike"] >= lo) & (working["Strike"] <= hi)]
       if not window.empty:
           working = window

       resist = working[
           (working["Strike"] >= spot) & (working["GEX_customer"] > 0)
       ].sort_values(["GEX_customer", "Strike"], ascending=[False, True]).head(top_n)

       support = working[
           (working["Strike"] <= spot) & (working["GEX_customer"] < 0)
       ].sort_values(["GEX_customer", "Strike"], ascending=[True, False]).head(top_n)

       if resist.empty:
           resist = working[working["GEX_customer"] > 0].sort_values(
               ["GEX_customer", "Strike"], ascending=[False, True]
           ).head(top_n)

       if support.empty:
           support = working[working["GEX_customer"] < 0].sort_values(
               ["GEX_customer", "Strike"], ascending=[True, False]
           ).head(top_n)

       return resist, support


def _build_focus_expiry_snapshot(df, spot, top_n=3, zone_pct=0.04):
       """
       Build a support / resistance view for the 2 nearest available expirations.

       Uses all options in the DataFrame (which is already filtered to the
       2 next expiring dates by the loader).
       """
       empty = {
           "expiry_label": "N/A",
           "dte": np.nan,
           "resist_zones": pd.DataFrame(),
           "support_zones": pd.DataFrame(),
       }

       if df is None or df.empty or "Expiration" not in df.columns:
           return empty

       work = df.copy()
       work["Expiration"] = pd.to_datetime(work["Expiration"], errors="coerce")
       work = work.dropna(subset=["Expiration", "Strike", "Tit."])
       if work.empty:
           return empty

       expiry_dates = sorted(work["Expiration"].dt.normalize().unique())
       if not expiry_dates:
           return empty

       today = pd.Timestamp.now().normalize()
       future_expiries = [d for d in expiry_dates if d >= today]
       if not future_expiries:
           future_expiries = expiry_dates[-2:] if len(expiry_dates) >= 2 else expiry_dates
       focus_expiries = future_expiries[:2]

       focus_df = work[work["Expiration"].dt.normalize().isin(focus_expiries)].copy()
       if focus_df.empty:
           return empty

       focus_df["GEX_abs"] = focus_df["Gamma"] * (spot ** 2) * focus_df["Tit."]

       call_frame = focus_df[
           focus_df["Tipo"].str.upper().str.contains("CALL")
       ].groupby("Strike", as_index=False).agg(GEX_customer=("GEX_abs", "sum"))

       put_frame = focus_df[
           focus_df["Tipo"].str.upper().str.contains("PUT")
       ].groupby("Strike", as_index=False).agg(GEX_customer=("GEX_abs", "sum"))
       put_frame["GEX_customer"] = -put_frame["GEX_customer"].abs()

       lo = spot * (1.0 - zone_pct)
       hi = spot * (1.0 + zone_pct)

       resist = call_frame[
           (call_frame["Strike"] >= spot) & (call_frame["Strike"] <= hi)
       ].sort_values(["GEX_customer", "Strike"], ascending=[False, True]).head(top_n)

       support = put_frame[
           (put_frame["Strike"] <= spot) & (put_frame["Strike"] >= lo)
       ].sort_values(["GEX_customer", "Strike"], ascending=[True, False]).head(top_n)

       if resist.empty:
           resist = call_frame.sort_values(["GEX_customer", "Strike"], ascending=[False, True]).head(top_n)
       if support.empty:
           support = put_frame.sort_values(["GEX_customer", "Strike"], ascending=[True, False]).head(top_n)

       dte_val = int(np.nanmin(focus_df["DTE"])) if "DTE" in focus_df.columns and not focus_df["DTE"].isna().all() else np.nan
       expiry_label = " + ".join(pd.Timestamp(d).strftime("%d/%m/%Y") for d in focus_expiries)

       return {
           "expiry_label": expiry_label,
           "dte": dte_val,
           "resist_zones": resist,
           "support_zones": support,
       }


def _format_oi(value):
       """Compact OI formatter for console tables."""
       if value is None or not np.isfinite(value):
           return "-"
       return f"{int(round(float(value))):,}"


def _strength_label(gex_value):
       """Simple strength bucket for GEX zones."""
       if gex_value is None or not np.isfinite(gex_value):
           return "N/A"
       gex_m = abs(float(gex_value)) / 1e6
       if gex_m >= 100:
           return "Strong"
       if gex_m >= 20:
           return "Mod"
       return "Weak"


def _build_pin_candidates_snapshot(df, spot, top_n=5, pct_range=0.05):
       """
       Build the top pin candidates near spot for the 2 nearest expirations.

       Dealer GEX is the inverse of customer GEX; high positive dealer GEX near
       spot tends to create a pinning effect as market makers hedge back toward
       those strikes.
       """
       empty = {
           "expiry_label": "N/A",
           "dte": np.nan,
           "pin_candidates": pd.DataFrame(),
       }

       if df is None or df.empty or "Expiration" not in df.columns or spot <= 0:
           return empty

       work = df.copy()
       work["Expiration"] = pd.to_datetime(work["Expiration"], errors="coerce")
       work = work.dropna(subset=["Expiration", "Strike", "Tit.", "Gamma"])
       if work.empty:
           return empty

       expiry_dates = sorted(work["Expiration"].dt.normalize().unique())
       if not expiry_dates:
           return empty

       today = pd.Timestamp.now().normalize()
       future_expiries = [d for d in expiry_dates if d >= today]
       if not future_expiries:
           future_expiries = expiry_dates[-2:] if len(expiry_dates) >= 2 else expiry_dates
       focus_expiries = future_expiries[:2]

       focus_df = work[work["Expiration"].dt.normalize().isin(focus_expiries)].copy()
       if focus_df.empty:
           return empty

       is_put = focus_df["Tipo"].str.upper().str.contains("PUT")
       sign = np.where(is_put, -1.0, 1.0)
       focus_df["GEX_customer"] = focus_df["Gamma"] * (spot ** 2) * focus_df["Tit."] * sign
       focus_df["dealer_gex"] = -focus_df["GEX_customer"]
       focus_df["call_oi"] = np.where(~is_put, focus_df["Tit."], 0.0)
       focus_df["put_oi"] = np.where(is_put, focus_df["Tit."], 0.0)

       lo = spot * (1.0 - pct_range)
       hi = spot * (1.0 + pct_range)
       near = focus_df[(focus_df["Strike"] >= lo) & (focus_df["Strike"] <= hi)]
       if near.empty:
           near = focus_df

       pins = near.groupby("Strike", as_index=False).agg(
           dealer_gex=("dealer_gex", "sum"),
           call_oi=("call_oi", "sum"),
           put_oi=("put_oi", "sum"),
       )

       pins = pins[pins["dealer_gex"] > 0].sort_values(
           ["dealer_gex", "Strike"], ascending=[False, False]
       ).head(top_n)

       if pins.empty:
           pins = near.groupby("Strike", as_index=False).agg(
               dealer_gex=("dealer_gex", "sum"),
               call_oi=("call_oi", "sum"),
               put_oi=("put_oi", "sum"),
           ).sort_values(["dealer_gex", "Strike"], ascending=[False, False]).head(top_n)

       dte_val = int(np.nanmin(focus_df["DTE"])) if "DTE" in focus_df.columns and not focus_df["DTE"].isna().all() else np.nan
       expiry_label = " + ".join(pd.Timestamp(d).strftime("%d/%m/%Y") for d in focus_expiries)

       return {
           "expiry_label": expiry_label,
           "dte": dte_val,
           "pin_candidates": pins,
       }


def _export_gex_csv(underlying, spot, call_wall, put_wall, gamma_flip, regime,
                    weekly_results, pin_snapshot, resist_zones, support_zones,
                    win_mapper, trade_signal=None, flyagonal=None,
                    win_symbol=""):
       """Write GEX levels to MQL5/Files/GEX_<underlying>.csv for the MT5 indicator."""
       # Resolve MQL5/Files path relative to SCRIPT_DIR
       mql5_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
       files_dir = os.path.join(mql5_root, 'Files')
       os.makedirs(files_dir, exist_ok=True)
       csv_path = os.path.join(files_dir, f'GEX_{underlying}.csv')

       def _win(val):
           """Return WIN current symbol equivalent or empty string."""
           if win_mapper is not None and np.isfinite(val):
               return f"{win_mapper.bova11_to_ind(val):.0f}"
           return ""

       rows = []
       rows.append(("spot", f"{spot:.4f}", _win(spot), ""))
       rows.append(("call_wall", f"{call_wall:.4f}" if np.isfinite(call_wall) else "", _win(call_wall) if np.isfinite(call_wall) else "", ""))
       rows.append(("put_wall", f"{put_wall:.4f}" if np.isfinite(put_wall) else "", _win(put_wall) if np.isfinite(put_wall) else "", ""))
       rows.append(("gamma_flip", f"{gamma_flip:.4f}" if np.isfinite(gamma_flip) else "", _win(gamma_flip) if np.isfinite(gamma_flip) else "", ""))
       rows.append(("regime", "1" if "POSITIVE" in regime else ("-1" if "NEGATIVE" in regime else "0"), "", ""))

       # Per-week walls
       for wk in weekly_results:
           if wk['gex_by_strike'].empty:
               continue
           tag = wk['label'].lower().replace(' ', '_')  # current_week / next_week
           exp = wk['friday_str']
           cw = wk['call_wall']
           pw = wk['put_wall']
           fl = wk['gamma_flip']
           rows.append((f"{tag}_call_wall", f"{cw:.4f}" if np.isfinite(cw) else "", _win(cw) if np.isfinite(cw) else "", exp))
           rows.append((f"{tag}_put_wall", f"{pw:.4f}" if np.isfinite(pw) else "", _win(pw) if np.isfinite(pw) else "", exp))
           rows.append((f"{tag}_flip", f"{fl:.4f}" if np.isfinite(fl) else "", _win(fl) if np.isfinite(fl) else "", exp))

       # Pin candidates
       pins = pin_snapshot.get('pin_candidates', pd.DataFrame())
       if not pins.empty:
           for i, (_, r) in enumerate(pins.head(5).iterrows(), 1):
               rows.append((f"pin_{i}", f"{r['Strike']:.4f}", _win(r['Strike']), ""))

       # Resistance zones
       if not resist_zones.empty:
           for i, (_, r) in enumerate(resist_zones.head(3).iterrows(), 1):
               rows.append((f"resist_{i}", f"{r['Strike']:.4f}", _win(r['Strike']), ""))

       # Support zones
       if not support_zones.empty:
           for i, (_, r) in enumerate(support_zones.head(3).iterrows(), 1):
               rows.append((f"support_{i}", f"{r['Strike']:.4f}", _win(r['Strike']), ""))

       # Trade signal
       if trade_signal is not None:
           sig_map = {'BUY': '1', 'SELL': '-1', 'BREAKOUT_DOWN': '-2',
                      'BREAKOUT_UP': '2', 'NEUTRAL': '0'}
           rows.append(("signal", sig_map.get(trade_signal['signal'], '0'), "", ""))
           rows.append(("signal_name", trade_signal['signal'], "", ""))
           rows.append(("signal_strength", str(trade_signal['strength']), "", ""))
           rows.append(("signal_regime", trade_signal['regime'], "", ""))

       # Flyagonal strategy levels
       if flyagonal is not None:
           rows.append(("fly_center", f"{flyagonal['center_strike']:.4f}", _win(flyagonal['center_strike']), flyagonal['near_expiry']))
           rows.append(("fly_lower", f"{flyagonal['lower_strike']:.4f}", _win(flyagonal['lower_strike']), flyagonal['far_expiry']))
           rows.append(("fly_upper", f"{flyagonal['upper_strike']:.4f}", _win(flyagonal['upper_strike']), flyagonal['far_expiry']))
           rows.append(("fly_net_premium", f"{flyagonal['net_premium']:.4f}", "", ""))
           rows.append(("fly_suitability", flyagonal['suitability'], "", ""))

       # Entry lines — 1.5% proximity zone from walls (matches generate_gex_trade_signals)
       proximity_pct = 0.015
       if np.isfinite(call_wall) and call_wall > 0:
           entry_sell = call_wall * (1.0 - proximity_pct)
           rows.append(("entry_sell", f"{entry_sell:.4f}", _win(entry_sell), ""))
       if np.isfinite(put_wall) and put_wall > 0:
           entry_buy = put_wall * (1.0 + proximity_pct)
           rows.append(("entry_buy", f"{entry_buy:.4f}", _win(entry_buy), ""))

       # Current WIN futures symbol name
       if win_symbol:
           rows.append(("win_symbol", win_symbol, "", ""))

       with open(csv_path, 'w', newline='') as f:
           f.write("key,value,win,expiry\n")
           for key, val, win, exp in rows:
               f.write(f"{key},{val},{win},{exp}\n")

       print(f"\n[CSV] Exported -> {csv_path}")


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
       resist_zones, support_zones = _select_significant_zones(zone_source, spot, top_n=3, zone_pct=0.04)
       focus_snapshot = _build_focus_expiry_snapshot(df, spot, top_n=3, zone_pct=0.04)
       pin_snapshot = _build_pin_candidates_snapshot(df, spot, top_n=5, pct_range=0.05)
       if not focus_snapshot['resist_zones'].empty or not focus_snapshot['support_zones'].empty:
           resist_zones = focus_snapshot['resist_zones']
           support_zones = focus_snapshot['support_zones']

       flip_dist_pct = ((spot - gamma_flip) / gamma_flip * 100.0) if np.isfinite(gamma_flip) and gamma_flip != 0 else np.nan
       sentiment_pt = _classify_sentiment_from_pcr(pcr_global)
       hedging_state = _hedging_state(spot, gamma_flip)

       print("\n" + "="*75)
       print(f"GEX SNAPSHOT SUMMARY -- {underlying}")
       print("="*75)
       if win_mapper is not None:
           cw_win = f" (WIN$N {win_mapper.bova11_to_ind(call_wall):,.0f})" if np.isfinite(call_wall) else ""
           pw_win = f" (WIN$N {win_mapper.bova11_to_ind(put_wall):,.0f})" if np.isfinite(put_wall) else ""
           flip_win = f" (WIN$N {win_mapper.bova11_to_ind(gamma_flip):,.0f})" if np.isfinite(gamma_flip) else ""
           spot_win = f" (WIN$N {win_mapper.bova11_to_ind(spot):,.0f})"
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
       print(f"HEDGING    : {hedging_state}")
       if focus_snapshot['expiry_label'] != 'N/A':
           dte_label = f"{focus_snapshot['dte']} DTE" if np.isfinite(focus_snapshot['dte']) else "N/A"
           print(f"FOCUS EXP. : {focus_snapshot['expiry_label']} ({dte_label})")

       if not pin_snapshot['pin_candidates'].empty:
           pin_dte_label = f"{pin_snapshot['dte']} DTE" if np.isfinite(pin_snapshot['dte']) else "N/A"
           print("\nPIN CANDIDATES (+/-5% FROM SPOT) SNAPSHOT:")
           print(f"Expiry: {pin_snapshot['expiry_label']} ({pin_dte_label})")
           if win_mapper is not None:
               print(f"{'Strike':>10} {'WIN$N':>10} {'Dealer GEX':>14} {'Calls OI':>12} {'Puts OI':>12}")
               print("-" * 66)
           else:
               print(f"{'Strike':>10} {'Dealer GEX':>14} {'Calls OI':>12} {'Puts OI':>12}")
               print("-" * 54)
           for _, row in pin_snapshot['pin_candidates'].iterrows():
               win_str = f"{win_mapper.bova11_to_ind(row['Strike']):>10,.0f} " if win_mapper is not None else ""
               print(
                   f"{row['Strike']:>10.2f} "
                   f"{win_str}"
                   f"{_format_gex_compact(row['dealer_gex']):>14} "
                   f"{_format_oi(row['call_oi']):>12} "
                   f"{_format_oi(row['put_oi']):>12}"
               )

       print("\nZONAS SIGNIFICATIVAS GEX:")
       if win_mapper is not None:
           print(f"{'ZONA':<14} {'STRIKE':>10} {'WIN$N':>10} {'GEX':>14} {'STRENGTH':>10}")
           print("-" * 66)
       else:
           print(f"{'ZONA':<14} {'STRIKE':>10} {'GEX':>14} {'STRENGTH':>10}")
           print("-" * 54)

       if resist_zones.empty and support_zones.empty:
           print(f"{'N/A':<14} {'-':>10} {'-':>14} {'-':>10}")
       else:
           for _, row in resist_zones.iterrows():
               win_str = f" {win_mapper.bova11_to_ind(row['Strike']):>10,.0f}" if win_mapper is not None else ""
               print(f"{'RESISTENCIA':<14} {row['Strike']:>10.2f}{win_str} {_format_gex_compact(row['GEX_customer']):>14} {_strength_label(row['GEX_customer']):>10}")
           for _, row in support_zones.iterrows():
               win_str = f" {win_mapper.bova11_to_ind(row['Strike']):>10,.0f}" if win_mapper is not None else ""
               print(f"{'SUPORTE':<14} {row['Strike']:>10.2f}{win_str} {_format_gex_compact(row['GEX_customer']):>14} {_strength_label(row['GEX_customer']):>10}")

       print("\nTOP 3 RESISTANCE ZONES:")
       if resist_zones.empty:
           print("  N/A")
       else:
           for idx, (_, row) in enumerate(resist_zones.reset_index(drop=True).iterrows(), start=1):
               win_str = f" (WIN$N {win_mapper.bova11_to_ind(row['Strike']):,.0f})" if win_mapper is not None else ""
               print(f"  {idx}. Strike {row['Strike']:.2f}{win_str} | GEX {_format_gex_compact(row['GEX_customer'])}")

       print("\nTOP 3 SUPPORT ZONES:")
       if support_zones.empty:
           print("  N/A")
       else:
           for idx, (_, row) in enumerate(support_zones.reset_index(drop=True).iterrows(), start=1):
               win_str = f" (WIN$N {win_mapper.bova11_to_ind(row['Strike']):,.0f})" if win_mapper is not None else ""
               print(f"  {idx}. Strike {row['Strike']:.2f}{win_str} | GEX {_format_gex_compact(row['GEX_customer'])}")
   
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

       # ----------------------------------------------------------------
       # GEX TRADE SIGNAL — regime + wall proximity
       # ----------------------------------------------------------------
       trade_signal = generate_gex_trade_signals(
           spot, gamma_flip, call_wall, put_wall
       )

       signal_colors = {
           'BUY': '',
           'SELL': '',
           'BREAKOUT_DOWN': '',
           'BREAKOUT_UP': '',
           'NEUTRAL': '',
       }
       RST = ''
       sc = signal_colors.get(trade_signal['signal'], '')
       strength_bar = '#' * trade_signal['strength'] + '.' * (3 - trade_signal['strength'])

       print(f"\n{'='*75}")
       print(f"GEX TRADE SIGNAL -- {underlying}")
       print(f"{'='*75}")
       print(f"  SIGNAL   : {sc}{trade_signal['signal']}{RST}  [{strength_bar}]")
       print(f"  REGIME   : {trade_signal['regime']}")
       print(f"  STRENGTH : {trade_signal['strength']}/3")
       print(f"  REASON   : {trade_signal['reason']}")
       print(f"{'='*75}")

       # ----------------------------------------------------------------
       # GEX ENTRY LEVELS — compute proximity zone for monitoring
       # ----------------------------------------------------------------
       proximity_pct = 0.015
       entry_sell = call_wall * (1.0 - proximity_pct) if np.isfinite(call_wall) and call_wall > 0 else np.nan
       entry_buy  = put_wall * (1.0 + proximity_pct)  if np.isfinite(put_wall) and put_wall > 0 else np.nan

       if win_mapper is not None:
           _eb = f" (WIN {win_mapper.bova11_to_ind(entry_buy):.0f})" if np.isfinite(entry_buy) else ""
           _es = f" (WIN {win_mapper.bova11_to_ind(entry_sell):.0f})" if np.isfinite(entry_sell) else ""
       else:
           _eb = _es = ""
       print(f"\n  ENTRY BUY  : {f'{entry_buy:.2f}' if np.isfinite(entry_buy) else 'N/A'}{_eb}")
       print(f"  ENTRY SELL : {f'{entry_sell:.2f}' if np.isfinite(entry_sell) else 'N/A'}{_es}")

       # ----------------------------------------------------------------
       # FLYAGONAL STRATEGY — Diagonal Butterfly from GEX levels
       # ----------------------------------------------------------------
       flyagonal = build_flyagonal(
           df, spot, weekly_results, pin_snapshot, regime,
           option_type='call',
       )
       print("\n" + format_flyagonal_snapshot(flyagonal, win_mapper=win_mapper))

       # ----------------------------------------------------------------
       # Export CSV for MQL5 indicator  →  MQL5/Files/GEX_<underlying>.csv
       # ----------------------------------------------------------------
       _export_gex_csv(
           underlying, spot, call_wall, put_wall, gamma_flip, regime,
           weekly_results, pin_snapshot, resist_zones, support_zones,
           win_mapper,
           trade_signal=trade_signal,
           flyagonal=flyagonal,
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

       # Return key GEX levels for the monitor
       return {
           'call_wall': call_wall,
           'put_wall': put_wall,
           'gamma_flip': gamma_flip,
           'regime': regime,
           'trade_signal': trade_signal,
       }


async def _monitor_gex_entries(mt5_conn, win_symbol, win_mapper,
                                call_wall, put_wall, gamma_flip):
    """
    Real-time spot price monitor for GEX entry execution.

    Polls WIN futures tick every GEX_MONITOR_INTERVAL seconds, re-evaluates
    the trade signal with the current spot, and sends a market order when
    a wall entry is triggered with sufficient signal strength.

    Stops when both BUY and SELL sides have been executed, or on KeyboardInterrupt.
    """
    import MetaTrader5 as _mt5
    from datetime import datetime as _dt

    proximity_pct = 0.015
    entry_buy_bova  = put_wall * (1.0 + proximity_pct) if np.isfinite(put_wall) and put_wall > 0 else np.nan
    entry_sell_bova = call_wall * (1.0 - proximity_pct) if np.isfinite(call_wall) and call_wall > 0 else np.nan

    win_entry_buy  = win_mapper.bova11_to_ind(entry_buy_bova) if np.isfinite(entry_buy_bova) else np.nan
    win_entry_sell = win_mapper.bova11_to_ind(entry_sell_bova) if np.isfinite(entry_sell_bova) else np.nan

    buy_executed  = False
    sell_executed = False

    print(f"\n{'='*75}")
    print(f"GEX MONITOR STARTED — {win_symbol}")
    print(f"{'='*75}")
    print(f"  Call Wall  : {call_wall:.2f} (BOVA) → {win_mapper.bova11_to_ind(call_wall):.0f} (WIN)" if np.isfinite(call_wall) else "  Call Wall  : N/A")
    print(f"  Put Wall   : {put_wall:.2f} (BOVA) → {win_mapper.bova11_to_ind(put_wall):.0f} (WIN)" if np.isfinite(put_wall) else "  Put Wall   : N/A")
    print(f"  Gamma Flip : {gamma_flip:.2f} (BOVA) → {win_mapper.bova11_to_ind(gamma_flip):.0f} (WIN)" if np.isfinite(gamma_flip) else "  Gamma Flip : N/A")
    print(f"  Entry BUY  : {entry_buy_bova:.2f} (BOVA) → {win_entry_buy:.0f} (WIN)" if np.isfinite(entry_buy_bova) else "  Entry BUY  : N/A")
    print(f"  Entry SELL : {entry_sell_bova:.2f} (BOVA) → {win_entry_sell:.0f} (WIN)" if np.isfinite(entry_sell_bova) else "  Entry SELL : N/A")
    print(f"  Volume     : {GEX_ORDER_VOLUME}")
    print(f"  Interval   : {GEX_MONITOR_INTERVAL}s")
    print(f"  Min Strength: {GEX_MIN_SIGNAL_STRENGTH}/3")
    print(f"{'='*75}")
    print(f"  Press Ctrl+C to stop monitoring.\n")

    tick_count = 0
    try:
        while not (buy_executed and sell_executed):
            tick = _mt5.symbol_info_tick(win_symbol)
            if tick is None:
                print(f"[GEX Monitor] Could not get tick for {win_symbol}")
                await asyncio.sleep(GEX_MONITOR_INTERVAL)
                continue

            win_spot = (tick.bid + tick.ask) / 2.0
            bova_spot = win_mapper.ind_to_bova11(win_spot)

            signal = generate_gex_trade_signals(bova_spot, gamma_flip, call_wall, put_wall)
            sig = signal['signal']
            strength = signal['strength']
            tick_count += 1

            # Log every tick (10s) with compact status
            ts = _dt.now().strftime("%H:%M:%S")
            side_status = (("BUY:DONE" if buy_executed else "BUY:wait") + " | "
                           + ("SELL:DONE" if sell_executed else "SELL:wait"))
            print(f"[{ts}] {win_symbol} {win_spot:.0f} | BOVA {bova_spot:.2f} | "
                  f"{sig} [{strength}/3] | {signal['regime']} | {side_status}")

            # --- Execute BUY ---
            if (sig == 'BUY' and strength >= GEX_MIN_SIGNAL_STRENGTH
                    and not buy_executed and np.isfinite(win_entry_buy)):
                print(f"\n[GEX Monitor] *** BUY TRIGGERED @ {tick.ask:.0f} ***")
                result = mt5_conn.place_order(
                    symbol=win_symbol,
                    order_type=_mt5.ORDER_TYPE_BUY,
                    volume=GEX_ORDER_VOLUME,
                    price=tick.ask,
                    deviation=GEX_ORDER_DEVIATION,
                    comment=f"GEX BUY PutWall {put_wall:.2f}",
                    magic=GEX_MAGIC_NUMBER,
                )
                if result is not None and hasattr(result, 'retcode') and result.retcode == _mt5.TRADE_RETCODE_DONE:
                    buy_executed = True
                    print(f"[GEX Monitor] BUY FILLED — order #{result.order}\n")
                else:
                    print(f"[GEX Monitor] BUY order sent (check logs above for status)\n")
                    buy_executed = True  # Avoid repeated attempts on same tick

            # --- Execute SELL ---
            elif (sig == 'SELL' and strength >= GEX_MIN_SIGNAL_STRENGTH
                      and not sell_executed and np.isfinite(win_entry_sell)):
                print(f"\n[GEX Monitor] *** SELL TRIGGERED @ {tick.bid:.0f} ***")
                result = mt5_conn.place_order(
                    symbol=win_symbol,
                    order_type=_mt5.ORDER_TYPE_SELL,
                    volume=GEX_ORDER_VOLUME,
                    price=tick.bid,
                    deviation=GEX_ORDER_DEVIATION,
                    comment=f"GEX SELL CallWall {call_wall:.2f}",
                    magic=GEX_MAGIC_NUMBER,
                )
                if result is not None and hasattr(result, 'retcode') and result.retcode == _mt5.TRADE_RETCODE_DONE:
                    sell_executed = True
                    print(f"[GEX Monitor] SELL FILLED — order #{result.order}\n")
                else:
                    print(f"[GEX Monitor] SELL order sent (check logs above for status)\n")
                    sell_executed = True

            await asyncio.sleep(GEX_MONITOR_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n[GEX Monitor] Stopped by user after {tick_count} ticks.")

    print(f"[GEX Monitor] Session ended. BUY={'DONE' if buy_executed else 'PENDING'} | "
          f"SELL={'DONE' if sell_executed else 'PENDING'}")


async def main():
    mt5_conn = MT5Connector()

    # Build DI1 term-structure once (spline-interpolated per expiry)
    build_di1_curve(mt5_conn)

    # Build Kalman mapper WIN$N <-> BOVA11 on 15-min bars (best for intraday)
    win_mapper = None
    win_symbol = ""
    if "BOVA11" in ASSET_SYMBOL:
        # Resolve the current WIN mini futures contract (e.g. WINM26)
        try:
            _exp_time, win_symbol = mt5_conn.get_symbol_futures("*WIN*")
            print(f"[i] Current WIN futures contract: {win_symbol}")
        except Exception as e:
            print(f"[!] Could not resolve current WIN symbol: {e}")
            win_symbol = ""

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

    bova11_gex = None
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
        win_sym_for_asset = win_symbol if asset == "BOVA11" else ""
        result = await analyze_options(
            spot_price,
            asset,
            win_mapper=mapper_for_asset,
            win_symbol=win_sym_for_asset,
            mt5_conn=mt5_conn,
        )
        if asset == "BOVA11" and result is not None:
            bova11_gex = result

    # --- Start real-time GEX monitor on WIN futures ---
    if (GEX_MONITOR_ENABLED and GEX_SEND_ORDERS
            and bova11_gex is not None
            and win_mapper is not None and win_symbol):
        await _monitor_gex_entries(
            mt5_conn, win_symbol, win_mapper,
            call_wall=bova11_gex['call_wall'],
            put_wall=bova11_gex['put_wall'],
            gamma_flip=bova11_gex['gamma_flip'],
        )
    elif GEX_MONITOR_ENABLED:
        reasons = []
        if not GEX_SEND_ORDERS:
            reasons.append("GEX_SEND_ORDERS is False")
        if bova11_gex is None:
            reasons.append("BOVA11 analysis failed or not in ASSET_SYMBOL")
        if win_mapper is None:
            reasons.append("WIN-BOVA11 mapper unavailable")
        if not win_symbol:
            reasons.append("WIN symbol not resolved")
        print(f"\n[GEX Monitor] Cannot start: {'; '.join(reasons)}")

asyncio.run(main())
# -*- coding: utf-8 -*-
"""
GEX Zones & Formatting Helpers
------------------------------
Pure utilities (no I/O, no print) used by main.py / gex_monitor.py:
  - sentiment / hedging / neutral-setup classifiers
  - significant zones, focus-expiry snapshot, pin candidates
  - nearest support/resistance lookup with pin fallback
  - apply_proximity_offset — the entry offset rule (single source of truth)
  - small console formatters (compact GEX, OI, strength)
"""
import numpy as np
import pandas as pd

from constants import GEX_WALL_PROXIMITY_PCT


# ----------------------------------------------------------------------
# Classifiers
# ----------------------------------------------------------------------
def classify_sentiment_from_pcr(pcr_global):
    """Map PCR to a short sentiment label used in the console snapshot."""
    if pcr_global is None or not np.isfinite(pcr_global):
        return "N/A"
    if pcr_global < 0.90:
        return "ALTISTA"
    if pcr_global > 1.10:
        return "BAIXISTA"
    return "NEUTRO"


def hedging_state(spot, gamma_flip):
    """Return a simple hedging state label similar to the dashboard card."""
    if gamma_flip is None or not np.isfinite(gamma_flip) or gamma_flip == 0:
        return "N/A"
    dist_pct = abs((spot - gamma_flip) / gamma_flip) * 100.0
    if dist_pct <= 0.50:
        return "DAMPED"
    return "DAMPED" if spot >= gamma_flip else "AMPLIFIED"


def is_neutral_setup(spot, gamma_flip, call_wall, put_wall, max_flip_distance_pct):
    """Neutral setup: spot between walls and close to gamma flip."""
    if not np.isfinite(spot) or not np.isfinite(gamma_flip) or gamma_flip == 0:
        return False
    if np.isfinite(call_wall) and np.isfinite(put_wall):
        lo = min(call_wall, put_wall)
        hi = max(call_wall, put_wall)
        if not (lo <= spot <= hi):
            return False
    flip_dist = abs((spot - gamma_flip) / gamma_flip)
    return flip_dist <= max_flip_distance_pct


# ----------------------------------------------------------------------
# Formatters
# ----------------------------------------------------------------------
def format_gex_compact(value):
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


def format_oi(value):
    """Compact OI formatter for console tables."""
    if value is None or not np.isfinite(value):
        return "-"
    return f"{int(round(float(value))):,}"


def strength_label(gex_value):
    """Simple strength bucket for GEX zones."""
    if gex_value is None or not np.isfinite(gex_value):
        return "N/A"
    gex_m = abs(float(gex_value)) / 1e6
    if gex_m >= 100:
        return "Strong"
    if gex_m >= 20:
        return "Mod"
    return "Weak"


# ----------------------------------------------------------------------
# Entry proximity offset — single source of truth
# ----------------------------------------------------------------------
def apply_proximity_offset(price, side, proximity_pct=None):
    """
    Apply the GEX_WALL_PROXIMITY_PCT entry offset to a wall/zone price.

    Sign convention (matches generate_gex_trade_signals):
      +pct → BUY  zone BELOW support       (offset = -|pct|)
             SELL zone ABOVE resistance    (offset = +|pct|)
      -pct → BUY  zone ABOVE support       (offset = +|pct|)
             SELL zone BELOW resistance    (offset = -|pct|)
    """
    if not np.isfinite(price):
        return price
    if proximity_pct is None:
        proximity_pct = GEX_WALL_PROXIMITY_PCT
    abs_p = abs(proximity_pct)
    if side == 'buy':
        offset = -abs_p if proximity_pct >= 0 else abs_p
    elif side == 'sell':
        offset = abs_p if proximity_pct >= 0 else -abs_p
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return price * (1.0 + offset)


# ----------------------------------------------------------------------
# Zone selection
# ----------------------------------------------------------------------
def select_significant_zones(gex_frame, spot, top_n=3, zone_pct=0.04):
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


def nearest_support_resistance(spot, support_zones, resist_zones,
                                put_wall=np.nan, call_wall=np.nan,
                                pin_candidates=None):
    """Return the closest support strike below spot and closest resistance strike above spot.
    Also considers pin candidate strikes (high dealer GEX = pinning magnets).
    Falls back to put_wall / call_wall when no S/R zone is found."""
    below_strikes = []
    above_strikes = []
    if not support_zones.empty:
        below = support_zones[support_zones['Strike'] <= spot]
        if not below.empty:
            below_strikes.extend(below['Strike'].tolist())
    if not resist_zones.empty:
        above = resist_zones[resist_zones['Strike'] >= spot]
        if not above.empty:
            above_strikes.extend(above['Strike'].tolist())
    if pin_candidates is not None and not pin_candidates.empty:
        pin_below = pin_candidates[pin_candidates['Strike'] <= spot]
        pin_above = pin_candidates[pin_candidates['Strike'] >= spot]
        if not pin_below.empty:
            below_strikes.extend(pin_below['Strike'].tolist())
        if not pin_above.empty:
            above_strikes.extend(pin_above['Strike'].tolist())

    entry_buy = float(max(below_strikes)) if below_strikes else np.nan
    entry_sell = float(min(above_strikes)) if above_strikes else np.nan

    if not np.isfinite(entry_buy) and np.isfinite(put_wall):
        entry_buy = put_wall
    if not np.isfinite(entry_sell) and np.isfinite(call_wall):
        entry_sell = call_wall
    return entry_buy, entry_sell


# ----------------------------------------------------------------------
# Snapshots
# ----------------------------------------------------------------------
def build_focus_expiry_snapshot(df, spot, top_n=3, zone_pct=0.04):
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


def build_pin_candidates_snapshot(df, spot, top_n=5, pct_range=0.05):
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

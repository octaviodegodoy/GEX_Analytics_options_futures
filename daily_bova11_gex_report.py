
import math
import logging
import os
import sys
from datetime import datetime

try:
    import MetaTrader5 as mt5
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as exc:
    missing = getattr(exc, "name", None) or str(exc)
    raise SystemExit(
        "Missing required dependency: {missing}. "
        "Install packages: MetaTrader5, numpy, pandas, matplotlib."
        .format(missing=missing)
    ) from exc

import b3_options_loader
import get_b3_data
from gex_utils import compute_weekly_walls, find_gamma_flip
from gex_zones import select_significant_zones as _select_significant_zones
from mt5_connector import MT5Connector

# --- LOGGING SETUP ---
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"gex_report_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _resolve_bova11_spot(mt5_conn: MT5Connector) -> float:
    info = mt5.symbol_info("BOVA11")
    if info is not None:
        bid = float(getattr(info, "bid", 0.0) or 0.0)
        ask = float(getattr(info, "ask", 0.0) or 0.0)
        last = float(getattr(info, "last", 0.0) or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if last > 0:
            return last

    daily = mt5_conn.get_data("BOVA11", mt5.TIMEFRAME_D1, 1, 0)
    if daily is not None and not daily.empty:
        return float(daily.iloc[-1]["close"])
    return float("nan")


def _multiday_only_open_interest(underlying="BOVA11", oi_csv_path=None, multiday_days=5, spot=None, tickers=None):
    multiday = get_b3_data.fetch_multiday_volume(underlying, num_days=multiday_days)
    if multiday.empty:
        return pd.DataFrame(columns=["ticker", "oi", "oi_source"])

    result = multiday[["ticker", "accumulated_volume"]].copy()
    result = result.rename(columns={"accumulated_volume": "oi"})
    result["oi_source"] = f"multiday_volume_{multiday_days}d"
    return result


def _build_top_strikes(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    work = df.copy()
    work["is_put"] = work["Tipo"].str.upper().str.contains("PUT")
    work["call_oi"] = np.where(~work["is_put"], work["Tit."], 0.0)
    work["put_oi"] = np.where(work["is_put"], work["Tit."], 0.0)

    top = work.groupby("Strike", as_index=False).agg(
        GEX_customer=("GEX_customer", "sum"),
        call_oi=("call_oi", "sum"),
        put_oi=("put_oi", "sum"),
    )
    top["abs_gex"] = top["GEX_customer"].abs()
    top["bias"] = np.where(top["GEX_customer"] >= 0, "CALL", "PUT")
    top = top.sort_values(["abs_gex", "Strike"], ascending=[False, True]).head(top_n)
    return top[["Strike", "GEX_customer", "bias", "call_oi", "put_oi"]]


def _compute_report(spot: float) -> dict:
    oi_mode = "B3_HISTORY"
    df = b3_options_loader.load_b3_options_data("BOVA11", spot)

    if df.empty:
        logger.error(f"No BOVA11 options data returned for mode {oi_mode}")
        raise RuntimeError(f"No BOVA11 options data returned for mode {oi_mode}")

    # --- DATA QUALITY CHECKS ---
    warnings = []
    # 1. Check for missing strikes (large gaps)
    strikes = sorted(df['Strike'].unique())
    if len(strikes) > 1:
        gaps = [b - a for a, b in zip(strikes[:-1], strikes[1:])]
        max_gap = max(gaps)
        if max_gap > 2 * np.median(gaps):
            warnings.append(f"WARNING: Large gap detected in strikes (max gap: {max_gap:.2f})")

    # 2. Check for stale data (no options with today's expiration or recent business day)
    today = pd.Timestamp.now().normalize()
    if not any(pd.to_datetime(df['Expiration']).dt.normalize() >= today):
        warnings.append("WARNING: No options with expiration >= today (data may be stale)")

    # 3. Check for suspicious OI/volume (all zeros or extreme outliers)
    if (df['Tit.'] == 0).all():
        warnings.append("WARNING: All open interest (Tit.) values are zero!")
    if (df['VolFin'] == 0).all():
        warnings.append("WARNING: All volume (VolFin) values are zero!")
    if df['Tit.'].max() > 10 * (df['Tit.'].median() if df['Tit.'].median() > 0 else 1):
        warnings.append("WARNING: Extreme OI outlier detected (max much greater than median)")
    if df['VolFin'].max() > 10 * (df['VolFin'].median() if df['VolFin'].median() > 0 else 1):
        warnings.append("WARNING: Extreme volume outlier detected (max much greater than median)")

    # 4. Check for low number of strikes or contracts
    if len(strikes) < 5:
        warnings.append(f"WARNING: Very few strikes found ({len(strikes)})")
    if len(df) < 10:
        warnings.append(f"WARNING: Very few option contracts found ({len(df)})")

    sign = np.where(df["Tipo"].str.upper().str.contains("PUT"), -1.0, 1.0)
    df["GEX_customer"] = df["Gamma"] * (spot ** 2) * df["Tit."] * sign

    weekly = compute_weekly_walls(df, spot)
    combined_parts = [wk["gex_by_strike"] for wk in weekly if not wk["gex_by_strike"].empty]
    if combined_parts:
        combined = pd.concat(combined_parts, ignore_index=True)
        combined = combined.groupby("Strike", as_index=False)["GEX_customer"].sum().sort_values("Strike")
    else:
        combined = df.groupby("Strike", as_index=False)["GEX_customer"].sum().sort_values("Strike")

    above = combined[combined["Strike"] >= spot]
    below = combined[combined["Strike"] <= spot]
    call_wall = float(above.loc[above["GEX_customer"].idxmax(), "Strike"]) if not above.empty else float("nan")
    put_wall = float(below.loc[below["GEX_customer"].abs().idxmax(), "Strike"]) if not below.empty else float("nan")
    gamma_flip = float(find_gamma_flip(df, spot))

    resist_zones, support_zones = _select_significant_zones(combined, spot, top_n=3, zone_pct=0.04)
    top_strikes = _build_top_strikes(df, top_n=10)

    calls = df[df["Tipo"].str.upper().str.contains("CALL")]
    puts = df[df["Tipo"].str.upper().str.contains("PUT")]
    total_call_oi = float(calls["Tit."].sum())
    total_put_oi = float(puts["Tit."].sum())
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else float("nan")

    peak_idx = combined["GEX_customer"].abs().idxmax()
    peak_strike = float(combined.loc[peak_idx, "Strike"])
    peak_gex = float(combined.loc[peak_idx, "GEX_customer"])

    return {
        "oi_mode": oi_mode,
        "spot": float(spot),
        "chain_rows": int(len(df)),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "total_gex": float(combined["GEX_customer"].sum()),
        "peak_strike": peak_strike,
        "peak_gex": peak_gex,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "pcr": pcr,
        "weekly": weekly,
        "resist_zones": resist_zones,
        "support_zones": support_zones,
        "top_strikes": top_strikes,
        "warnings": warnings,
    }


def _fmt_num(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:,.2f}"


def _fmt_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _print_report(report: dict) -> None:
    logger.info(f"{'=' * 90}")
    logger.info(f"BOVA11 DAILY GEX REPORT -- {report['oi_mode']}")
    logger.info(f"{'=' * 90}")
    logger.info(f"Spot           : {_fmt_num(report['spot'])}")
    logger.info(f"Chain Rows     : {report['chain_rows']}")
    logger.info(f"Call Wall      : {_fmt_num(report['call_wall'])}")
    logger.info(f"Put Wall       : {_fmt_num(report['put_wall'])}")
    logger.info(f"Gamma Flip     : {_fmt_num(report['gamma_flip'])}")
    logger.info(f"Total GEX      : {_fmt_num(report['total_gex'])}")
    logger.info(f"Peak Strike    : {_fmt_num(report['peak_strike'])}")
    logger.info(f"Peak GEX       : {_fmt_num(report['peak_gex'])}")
    logger.info(f"Total Call OI  : {_fmt_num(report['total_call_oi'])}")
    logger.info(f"Total Put OI   : {_fmt_num(report['total_put_oi'])}")
    logger.info(f"PCR            : {_fmt_pct(report['pcr'])}")
    # Print any data quality warnings
    if 'warnings' in report and report['warnings']:
        logger.warning("DATA QUALITY WARNINGS:")
        for w in report['warnings']:
            logger.warning(f"- {w}")

    # --- GEX BY STRIKE PLOT ---
    try:
        strikes = report['top_strikes']['Strike'].values if not report['top_strikes'].empty else []
        gex = report['top_strikes']['GEX_customer'].values if not report['top_strikes'].empty else []
        if len(strikes) > 0 and len(gex) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(strikes, gex, color=np.where(np.array(gex) > 0, 'royalblue', 'tomato'))
            plt.axhline(0, color='black', linewidth=0.8)
            if np.isfinite(report.get('gamma_flip', float('nan'))):
                plt.axvline(report['gamma_flip'], color='purple', linestyle='--', label='Gamma Flip')
            if np.isfinite(report.get('call_wall', float('nan'))):
                plt.axvline(report['call_wall'], color='green', linestyle=':', label='Call Wall')
            if np.isfinite(report.get('put_wall', float('nan'))):
                plt.axvline(report['put_wall'], color='red', linestyle=':', label='Put Wall')
            plt.title(f"BOVA11 GEX by Strike ({report['oi_mode']})")
            plt.xlabel('Strike')
            plt.ylabel('Customer GEX')
            plt.legend()
            plt.tight_layout()
            plot_path = f"bova11_gex_by_strike_{report['oi_mode'].lower()}.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"GEX by strike plot saved as {plot_path}")
    except Exception as e:
        logger.error(f"[Plotting Error] {e}")

    print("\nTOP CUSTOMER GEX STRIKES")
    print(report["top_strikes"].to_string(index=False, formatters={
        "Strike": lambda v: f"{float(v):.2f}",
        "GEX_customer": lambda v: f"{float(v):,.2f}",
        "call_oi": lambda v: f"{float(v):,.0f}",
        "put_oi": lambda v: f"{float(v):,.0f}",
    }))

    print("\nWEEKLY WALLS")
    for wk in report["weekly"]:
        print(
            f"- {wk['label']} {wk['friday_str']} | DTE {wk['dte']} | "
            f"CW {_fmt_num(wk['call_wall'])} | PW {_fmt_num(wk['put_wall'])} | "
            f"Flip {_fmt_num(wk['gamma_flip'])} | Total GEX {_fmt_num(wk['total_gex'])}"
        )

    print("\nRESISTANCE ZONES")
    if report["resist_zones"] is not None and not report["resist_zones"].empty:
        print(report["resist_zones"][["Strike", "GEX_customer"]].to_string(index=False, formatters={
            "Strike": lambda v: f"{float(v):.2f}",
            "GEX_customer": lambda v: f"{float(v):,.2f}",
        }))
    else:
        print("NONE")

    print("\nSUPPORT ZONES")
    if report["support_zones"] is not None and not report["support_zones"].empty:
        print(report["support_zones"][["Strike", "GEX_customer"]].to_string(index=False, formatters={
            "Strike": lambda v: f"{float(v):.2f}",
            "GEX_customer": lambda v: f"{float(v):,.2f}",
        }))
    else:
        print("NONE")


def main() -> None:
    mt5_conn = MT5Connector()
    try:
        spot = _resolve_bova11_spot(mt5_conn)
        if not np.isfinite(spot) or spot <= 0:
            logger.error("Could not resolve BOVA11 spot price")
            raise RuntimeError("Could not resolve BOVA11 spot price")

        report = _compute_report(spot)
        _print_report(report)
    except Exception as e:
        logger.exception(f"[FATAL ERROR] {e}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
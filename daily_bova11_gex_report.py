import math
from contextlib import contextmanager

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import b3_options_loader
import get_b3_data
from gex_utils import compute_weekly_walls, find_gamma_flip
from main import _select_significant_zones
from mt5_connector import MT5Connector
from rtd_oi_reader import read_rtd_oi


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


@contextmanager
def _loader_open_interest_override(func):
    original = b3_options_loader.fetch_open_interest
    b3_options_loader.fetch_open_interest = func
    try:
        yield
    finally:
        b3_options_loader.fetch_open_interest = original


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


def _compute_report(spot: float, use_rtd: bool) -> dict:
    oi_mode = "RTD_CSV" if use_rtd else "B3_VOLUME_PROXY"
    override = b3_options_loader.fetch_open_interest if use_rtd else _multiday_only_open_interest

    with _loader_open_interest_override(override):
        df = b3_options_loader.load_b3_options_data("BOVA11", spot)

    if df.empty:
        raise RuntimeError(f"No BOVA11 options data returned for mode {oi_mode}")

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

    rtd_rows = 0
    rtd_sum = 0
    rtd_matched = 0
    if use_rtd:
        rtd_df = read_rtd_oi(spot=spot)
        rtd_rows = int(len(rtd_df))
        rtd_sum = int(rtd_df["oi"].sum()) if not rtd_df.empty else 0
        if not rtd_df.empty:
            merged = df[["Ticker"]].merge(
                rtd_df.rename(columns={"ticker": "Ticker", "oi": "RTD_OI"}),
                on="Ticker",
                how="left",
            )
            rtd_matched = int(merged["RTD_OI"].fillna(0).gt(0).sum())

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
        "rtd_rows": rtd_rows,
        "rtd_sum": rtd_sum,
        "rtd_matched": rtd_matched,
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
    print(f"\n{'=' * 90}")
    print(f"BOVA11 DAILY GEX REPORT -- {report['oi_mode']}")
    print(f"{'=' * 90}")
    print(f"Spot           : {_fmt_num(report['spot'])}")
    print(f"Chain Rows     : {report['chain_rows']}")
    print(f"Call Wall      : {_fmt_num(report['call_wall'])}")
    print(f"Put Wall       : {_fmt_num(report['put_wall'])}")
    print(f"Gamma Flip     : {_fmt_num(report['gamma_flip'])}")
    print(f"Total GEX      : {_fmt_num(report['total_gex'])}")
    print(f"Peak Strike    : {_fmt_num(report['peak_strike'])}")
    print(f"Peak GEX       : {_fmt_num(report['peak_gex'])}")
    print(f"Total Call OI  : {_fmt_num(report['total_call_oi'])}")
    print(f"Total Put OI   : {_fmt_num(report['total_put_oi'])}")
    print(f"PCR            : {_fmt_pct(report['pcr'])}")
    if report['oi_mode'] == 'RTD_CSV':
        print(f"RTD CSV Rows   : {report['rtd_rows']}")
        print(f"RTD OI Sum     : {report['rtd_sum']:,}")
        print(f"RTD Matched    : {report['rtd_matched']}")

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


def _print_comparison(with_rtd: dict, no_rtd: dict) -> None:
    print(f"\n{'=' * 90}")
    print("COMPARISON -- WITH RTD CSV OI VS B3 VOLUME PROXY")
    print(f"{'=' * 90}")

    rows = [
        ("Call Wall", with_rtd["call_wall"], no_rtd["call_wall"]),
        ("Put Wall", with_rtd["put_wall"], no_rtd["put_wall"]),
        ("Gamma Flip", with_rtd["gamma_flip"], no_rtd["gamma_flip"]),
        ("Total GEX", with_rtd["total_gex"], no_rtd["total_gex"]),
        ("Peak Strike", with_rtd["peak_strike"], no_rtd["peak_strike"]),
        ("Peak GEX", with_rtd["peak_gex"], no_rtd["peak_gex"]),
        ("Call OI", with_rtd["total_call_oi"], no_rtd["total_call_oi"]),
        ("Put OI", with_rtd["total_put_oi"], no_rtd["total_put_oi"]),
        ("PCR", with_rtd["pcr"], no_rtd["pcr"]),
    ]

    print(f"{'Metric':<14} {'With RTD':>18} {'No RTD':>18} {'Delta':>18}")
    for label, a, b in rows:
        delta = a - b if np.isfinite(a) and np.isfinite(b) else float("nan")
        print(f"{label:<14} {_fmt_num(a):>18} {_fmt_num(b):>18} {_fmt_num(delta):>18}")


def main() -> None:
    mt5_conn = MT5Connector()
    try:
        spot = _resolve_bova11_spot(mt5_conn)
        if not np.isfinite(spot) or spot <= 0:
            raise RuntimeError("Could not resolve BOVA11 spot price")

        with_rtd = _compute_report(spot, use_rtd=True)
        no_rtd = _compute_report(spot, use_rtd=False)

        _print_report(with_rtd)
        _print_report(no_rtd)
        _print_comparison(with_rtd, no_rtd)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
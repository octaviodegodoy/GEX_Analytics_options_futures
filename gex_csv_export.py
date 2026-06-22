# -*- coding: utf-8 -*-
"""
GEX CSV Export — writes MQL5/Files/GEX_<underlying>.csv consumed by GEX_Walls.mq5
"""
import os
import numpy as np
import pandas as pd

from gex_zones import nearest_support_resistance, apply_proximity_offset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def export_gex_csv(underlying, spot, call_wall, put_wall, gamma_flip, regime,
                   weekly_results, pin_snapshot, resist_zones, support_zones,
                   win_mapper, trade_signal=None, flyagonal=None,
                   strangle=None, win_symbol=""):
    """Write GEX levels to MQL5/Files/GEX_<underlying>.csv for the MT5 indicator."""
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

    # Strangle strategy levels
    if strangle is not None:
        rows.append(("strangle_call", f"{strangle['call_strike']:.4f}", _win(strangle['call_strike']), strangle['expiry']))
        rows.append(("strangle_put", f"{strangle['put_strike']:.4f}", _win(strangle['put_strike']), strangle['expiry']))
        rows.append(("strangle_direction", strangle['direction'], "", ""))
        rows.append(("strangle_net_premium", f"{strangle['net_premium']:.4f}", "", ""))
        rows.append(("strangle_suitability", strangle['suitability'], "", ""))

    # Entry lines — closest support / resistance to spot with directional offset
    _pin_df = pin_snapshot.get('pin_candidates', pd.DataFrame()) if pin_snapshot is not None else pd.DataFrame()
    entry_buy, entry_sell = nearest_support_resistance(
        spot, support_zones, resist_zones,
        put_wall=put_wall, call_wall=call_wall, pin_candidates=_pin_df,
    )
    entry_buy = apply_proximity_offset(entry_buy, 'buy')
    entry_sell = apply_proximity_offset(entry_sell, 'sell')
    if np.isfinite(entry_sell):
        rows.append(("entry_sell", f"{entry_sell:.4f}", _win(entry_sell), ""))
    if np.isfinite(entry_buy):
        rows.append(("entry_buy", f"{entry_buy:.4f}", _win(entry_buy), ""))

    # Current WIN futures symbol name
    if win_symbol:
        rows.append(("win_symbol", win_symbol, "", ""))

    with open(csv_path, 'w', newline='') as f:
        f.write("key,value,win,expiry\n")
        for key, val, win, exp in rows:
            f.write(f"{key},{val},{win},{exp}\n")

    print(f"\n[CSV] Exported -> {csv_path}")

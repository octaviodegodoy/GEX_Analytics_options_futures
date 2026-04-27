# -*- coding: utf-8 -*-
"""
GEX Live Trading Monitor
------------------------
Real-time WIN futures tick polling with:
  - Wall-entry triggers gated by signal strength + neutral setup + 5m confirmation
  - Margin-budgeted volume sizing with Fibonacci DCA
  - Trailing stop activated at fixed % of budget
  - Pre-trade & RTD-driven GEX level refresh
  - Daily realized loss cap

Extracted from main.py without behavior changes; the only refactor is
deduplicating the proximity offset via gex_zones.apply_proximity_offset.
"""
import asyncio
import math
import time as _time
from datetime import datetime as _dt

import numpy as np
import pandas as pd

import MetaTrader5 as _mt5

from constants import (
    GEX_ORDER_VOLUME, GEX_ORDER_DEVIATION, GEX_MIN_SIGNAL_STRENGTH,
    GEX_MONITOR_INTERVAL, GEX_MAGIC_NUMBER,
    GEX_MARGIN_FREE_PCT, GEX_SL_RISK_PCT, GEX_TRAILING_ACTIVATION_PCT,
    GEX_DCA_LOSS_STEP_PCT, GEX_DCA_MAX_ORDERS,
    GEX_RTD_REFRESH_INTERVAL,
    GEX_MIN_SL_POINTS, GEX_TRAILING_DISTANCE_FACTOR,
    GEX_MAX_DAILY_LOSS_PCT, GEX_TP_AT_OPPOSITE_WALL,
    GEX_TRADE_WINDOW_START, GEX_TRADE_WINDOW_END,
    GEX_PRE_TRADE_REFRESH_MIN,
    GEX_REQUIRE_5M_CONFIRMATION, GEX_CONFIRMATION_MINUTES,
    GEX_NEUTRAL_ONLY, GEX_NEUTRAL_MAX_FLIP_DISTANCE_PCT,
)
from gex_utils import generate_gex_trade_signals
from gex_zones import (
    nearest_support_resistance,
    is_neutral_setup,
    apply_proximity_offset,
)
from kalman_price_mapper import (
    build_ind_bova11_mapper,
    build_ind_bova11_mapper_intraday,
)
from mt5_connector import sanitize_modify_sl


async def monitor_gex_entries(mt5_conn, win_symbol, win_mapper,
                              call_wall, put_wall, gamma_flip,
                              support_zones=None, resist_zones=None,
                              pin_candidates=None):
    """
    Real-time spot price monitor for GEX entry execution.

    Polls WIN futures tick every GEX_MONITOR_INTERVAL seconds, re-evaluates
    the trade signal with the current spot, and sends a market order when
    a wall entry is triggered with sufficient signal strength.

    When RTD OI file is updated (Profit Pro export), GEX levels are
    recalculated automatically (controlled by GEX_RTD_REFRESH_INTERVAL).

    Stops when both BUY and SELL sides have been executed, or on KeyboardInterrupt.
    """
    # Late import to avoid circular dependency main.py <-> gex_monitor.py
    from main import analyze_options
    from rtd_oi_reader import rtd_data_changed

    def _refresh_win_contract(current_symbol, current_mapper, mt5c):
        """Re-resolve the WIN futures contract; rebuild mapper if it changed."""
        try:
            (_exp, new_symbol), expiring_sym = mt5c.get_symbol_futures(
                "*WIN*", include_expiring=True
            )
        except Exception as e:
            print(f"[WIN] Could not re-resolve contract: {e} — keeping {current_symbol}")
            return current_symbol, current_mapper

        if new_symbol != current_symbol:
            ts = _dt.now().strftime("%H:%M:%S")
            print(f"\n[{ts}] [WIN] Contract rolled: {current_symbol} → {new_symbol}")
            ind_syms = [new_symbol] if new_symbol else []
            if expiring_sym and expiring_sym not in ind_syms:
                ind_syms.append(expiring_sym)
            new_mapper = None
            for ind_sym in ind_syms:
                try:
                    new_mapper = build_ind_bova11_mapper_intraday(
                        mt5c, ind_symbol=ind_sym, bova11_symbol="BOVA11"
                    )
                    print(f"[{ts}] [WIN] Mapper rebuilt using {ind_sym}")
                    break
                except Exception:
                    try:
                        new_mapper = build_ind_bova11_mapper(
                            mt5c, ind_symbol=ind_sym, bova11_symbol="BOVA11"
                        )
                        print(f"[{ts}] [WIN] Mapper rebuilt (daily) using {ind_sym}")
                        break
                    except Exception:
                        pass
            if new_mapper is not None:
                return new_symbol, new_mapper
            else:
                print(f"[{ts}] [WIN] Mapper rebuild failed — keeping old mapper with new symbol")
                return new_symbol, current_mapper
        return current_symbol, current_mapper

    if support_zones is None:
        support_zones = pd.DataFrame()
    if resist_zones is None:
        resist_zones = pd.DataFrame()
    if pin_candidates is None:
        pin_candidates = pd.DataFrame()

    # Use nearest support/resistance zones for entry levels
    sym_info = _mt5.symbol_info("BOVA11")
    _init_spot = (sym_info.bid + sym_info.ask) / 2.0 if sym_info else 0.0
    entry_buy_bova, entry_sell_bova = nearest_support_resistance(
        _init_spot, support_zones, resist_zones,
        put_wall=put_wall, call_wall=call_wall, pin_candidates=pin_candidates,
    )
    entry_buy_bova = apply_proximity_offset(entry_buy_bova, 'buy')
    entry_sell_bova = apply_proximity_offset(entry_sell_bova, 'sell')

    win_entry_buy  = win_mapper.bova11_to_ind(entry_buy_bova) if np.isfinite(entry_buy_bova) else np.nan
    win_entry_sell = win_mapper.bova11_to_ind(entry_sell_bova) if np.isfinite(entry_sell_bova) else np.nan

    buy_executed  = False
    sell_executed = False

    # Per-side cooldown after a failed order_send so we don't hammer the
    # broker every tick when the signal stays up. Reset to 0 on success.
    _buy_retry_after  = 0.0  # epoch seconds; new BUY allowed when time.time() >= this
    _sell_retry_after = 0.0
    _ORDER_RETRY_COOLDOWN = 30  # seconds between failed-send retries

    # Trailing stop state per side
    trail_buy  = None
    trail_sell = None

    # --- Compute initial margin budget (needed by reconstructed trail states) ---
    _acct_init = _mt5.account_info()
    margin_budget = _acct_init.margin_free * GEX_MARGIN_FREE_PCT if _acct_init else 0.0

    # --- Detect existing GEX positions to avoid duplicates ---
    _existing = _mt5.positions_get(symbol=win_symbol)
    _sym_info = _mt5.symbol_info(win_symbol)
    _tick_val = _sym_info.trade_tick_value if _sym_info else 1.0
    _tick_sz  = _sym_info.trade_tick_size if _sym_info else 5.0

    for _p in (_existing or []):
        if _p.magic == GEX_MAGIC_NUMBER:
            if _p.type == _mt5.POSITION_TYPE_BUY:
                buy_executed = True
                _buy_positions = [p for p in _existing if p.magic == GEX_MAGIC_NUMBER and p.type == _mt5.POSITION_TYPE_BUY]
                _total_vol = sum(p.volume for p in _buy_positions)
                _avg_entry = sum(p.price_open * p.volume for p in _buy_positions) / _total_vol if _total_vol > 0 else _p.price_open
                _init_vol = _buy_positions[0].volume if _buy_positions else _p.volume
                _sl_price = _p.sl
                _sl_pts = abs(_avg_entry - _sl_price) if _sl_price > 0 else 0.0
                trail_buy = {
                    'entry': _avg_entry, 'vol': _init_vol,
                    'best': _avg_entry, 'active': False,
                    'tick_sz': _tick_sz, 'tick_val': _tick_val,
                    'sl_points': _sl_pts, 'sl_price': _sl_price,
                    'dca_count': max(len(_buy_positions) - 1, 0), 'wall': 'PutWall',
                    'margin_budget': margin_budget,
                }
                print(f"[GEX Monitor] Existing BUY position(s) detected — "
                      f"{len(_buy_positions)} pos, {_total_vol:.0f} vol, "
                      f"avg entry {_avg_entry:.0f}, DCA #{trail_buy['dca_count']}/{GEX_DCA_MAX_ORDERS}")
            elif _p.type == _mt5.POSITION_TYPE_SELL:
                sell_executed = True
                _sell_positions = [p for p in _existing if p.magic == GEX_MAGIC_NUMBER and p.type == _mt5.POSITION_TYPE_SELL]
                _total_vol = sum(p.volume for p in _sell_positions)
                _avg_entry = sum(p.price_open * p.volume for p in _sell_positions) / _total_vol if _total_vol > 0 else _p.price_open
                _init_vol = _sell_positions[0].volume if _sell_positions else _p.volume
                _sl_price = _p.sl
                _sl_pts = abs(_sl_price - _avg_entry) if _sl_price > 0 else 0.0
                trail_sell = {
                    'entry': _avg_entry, 'vol': _init_vol,
                    'best': _avg_entry, 'active': False,
                    'tick_sz': _tick_sz, 'tick_val': _tick_val,
                    'sl_points': _sl_pts, 'sl_price': _sl_price,
                    'dca_count': max(len(_sell_positions) - 1, 0), 'wall': 'CallWall',
                    'margin_budget': margin_budget,
                }
                print(f"[GEX Monitor] Existing SELL position(s) detected — "
                      f"{len(_sell_positions)} pos, {_total_vol:.0f} vol, "
                      f"avg entry {_avg_entry:.0f}, DCA #{trail_sell['dca_count']}/{GEX_DCA_MAX_ORDERS}")

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
    print(f"  Trade Window: {GEX_TRADE_WINDOW_START} – {GEX_TRADE_WINDOW_END}")
    print(f"  Pre-Trade Refresh: {GEX_PRE_TRADE_REFRESH_MIN}min before window")
    print(f"  Daily Loss Cap: {GEX_MAX_DAILY_LOSS_PCT:.0%} of budget")
    print(f"  Min SL Floor: {GEX_MIN_SL_POINTS} pts")
    print(f"  Trailing Activation: {GEX_TRAILING_ACTIVATION_PCT:.0%} of budget = R${margin_budget * GEX_TRAILING_ACTIVATION_PCT:,.2f}")
    print(f"  Trailing Factor: {GEX_TRAILING_DISTANCE_FACTOR:.0%} of SL")
    print(f"  TP at Opposite Wall: {'Yes' if GEX_TP_AT_OPPOSITE_WALL else 'No'}")
    print(f"  Budget     : R${margin_budget:,.2f} ({GEX_MARGIN_FREE_PCT:.1%} of margin_free)")
    print(f"  RTD Refresh : {'every ' + str(GEX_RTD_REFRESH_INTERVAL) + 's' if GEX_RTD_REFRESH_INTERVAL > 0 else 'disabled'}")
    _confirm_ticks = max(1, int(math.ceil((GEX_CONFIRMATION_MINUTES * 60) / max(GEX_MONITOR_INTERVAL, 1))))
    print(f"  5m Confirmation: {'On' if GEX_REQUIRE_5M_CONFIRMATION else 'Off'} ({_confirm_ticks} ticks)")
    print(f"  Neutral Setup Only: {'On' if GEX_NEUTRAL_ONLY else 'Off'} (<= {GEX_NEUTRAL_MAX_FLIP_DISTANCE_PCT:.2%} from flip)")
    print(f"{'='*75}")
    print(f"  Press Ctrl+C to stop monitoring.\n")

    # RTD OI refresh state
    _rtd_last_check = _time.monotonic()

    # Daily loss tracking — halt new entries when exceeded
    _daily_realized_pnl = 0.0
    _daily_loss_limit = margin_budget * GEX_MAX_DAILY_LOSS_PCT

    # TP levels (WIN prices) — opposite wall for each side
    win_tp_buy = win_mapper.bova11_to_ind(call_wall) if np.isfinite(call_wall) else np.nan
    win_tp_sell = win_mapper.bova11_to_ind(put_wall) if np.isfinite(put_wall) else np.nan

    # Pre-trade GEX refresh: recalculate levels N minutes before trading window
    _pre_trade_h, _pre_trade_m = map(int, GEX_TRADE_WINDOW_START.split(":"))
    _pre_trade_total_min = _pre_trade_h * 60 + _pre_trade_m - GEX_PRE_TRADE_REFRESH_MIN
    _pre_trade_time = f"{_pre_trade_total_min // 60:02d}:{_pre_trade_total_min % 60:02d}"
    _pre_trade_refreshed = False
    _waiting_msg_printed = False
    buy_confirm_ticks = 0
    sell_confirm_ticks = 0

    tick_count = 0
    try:
        while True:
            # --- Pre-trade GEX refresh: recalculate levels before trading window ---
            _now_hm = _dt.now().strftime("%H:%M")
            if (not _pre_trade_refreshed
                    and _pre_trade_time <= _now_hm < GEX_TRADE_WINDOW_START):
                _pre_trade_refreshed = True
                ts_now = _dt.now().strftime("%H:%M:%S")
                print(f"\n[{ts_now}] [PRE-TRADE] {GEX_PRE_TRADE_REFRESH_MIN}min before trading window — recalculating GEX levels...")
                try:
                    win_symbol, win_mapper = _refresh_win_contract(win_symbol, win_mapper, mt5_conn)

                    sym_info = _mt5.symbol_info("BOVA11")
                    _pt_spot = (sym_info.bid + sym_info.ask) / 2.0 if sym_info else 0.0
                    if _pt_spot > 0:
                        _pt_result = await analyze_options(
                            _pt_spot, "BOVA11",
                            win_mapper=win_mapper,
                            win_symbol=win_symbol,
                            mt5_conn=mt5_conn,
                        )
                        if _pt_result is not None:
                            old_cw, old_pw, old_gf = call_wall, put_wall, gamma_flip
                            call_wall = _pt_result['call_wall']
                            put_wall = _pt_result['put_wall']
                            gamma_flip = _pt_result['gamma_flip']

                            support_zones = _pt_result.get('support_zones', pd.DataFrame())
                            resist_zones = _pt_result.get('resist_zones', pd.DataFrame())
                            pin_candidates = _pt_result.get('pin_candidates', pd.DataFrame())
                            entry_buy_bova, entry_sell_bova = nearest_support_resistance(
                                _pt_spot, support_zones, resist_zones,
                                put_wall=put_wall, call_wall=call_wall, pin_candidates=pin_candidates,
                            )
                            entry_buy_bova = apply_proximity_offset(entry_buy_bova, 'buy')
                            entry_sell_bova = apply_proximity_offset(entry_sell_bova, 'sell')
                            win_entry_buy = win_mapper.bova11_to_ind(entry_buy_bova) if np.isfinite(entry_buy_bova) else np.nan
                            win_entry_sell = win_mapper.bova11_to_ind(entry_sell_bova) if np.isfinite(entry_sell_bova) else np.nan
                            win_tp_buy = win_mapper.bova11_to_ind(call_wall) if np.isfinite(call_wall) else np.nan
                            win_tp_sell = win_mapper.bova11_to_ind(put_wall) if np.isfinite(put_wall) else np.nan

                            print(f"[{ts_now}] [PRE-TRADE] GEX REFRESHED")
                            print(f"  Call Wall : {old_cw:.2f} → {call_wall:.2f}" if np.isfinite(call_wall) else f"  Call Wall : N/A")
                            print(f"  Put Wall  : {old_pw:.2f} → {put_wall:.2f}" if np.isfinite(put_wall) else f"  Put Wall  : N/A")
                            print(f"  Gamma Flip: {old_gf:.2f} → {gamma_flip:.2f}" if np.isfinite(gamma_flip) else f"  Gamma Flip: N/A")
                            print(f"  Entry BUY : {entry_buy_bova:.2f} → {win_entry_buy:.0f} (WIN)" if np.isfinite(entry_buy_bova) else f"  Entry BUY : N/A")
                            print(f"  Entry SELL: {entry_sell_bova:.2f} → {win_entry_sell:.0f} (WIN)" if np.isfinite(entry_sell_bova) else f"  Entry SELL: N/A")
                        else:
                            print(f"[{ts_now}] [PRE-TRADE] Reanalysis returned nothing — keeping old levels")
                    else:
                        print(f"[{ts_now}] [PRE-TRADE] BOVA11 spot unavailable — keeping old levels")
                except Exception as e:
                    ts_now = _dt.now().strftime("%H:%M:%S")
                    print(f"[{ts_now}] [PRE-TRADE] Refresh failed: {e} — keeping old levels")

            # --- RTD OI refresh: recalculate GEX when Profit Pro updates the file ---
            if GEX_RTD_REFRESH_INTERVAL > 0:
                now_mono = _time.monotonic()
                if now_mono - _rtd_last_check >= GEX_RTD_REFRESH_INTERVAL:
                    _rtd_last_check = now_mono
                    if rtd_data_changed():
                        ts_now = _dt.now().strftime("%H:%M:%S")
                        print(f"\n[{ts_now}] [RTD] OI file updated — recalculating GEX levels...")
                        try:
                            win_symbol, win_mapper = _refresh_win_contract(win_symbol, win_mapper, mt5_conn)

                            sym_info = _mt5.symbol_info("BOVA11")
                            rtd_spot = (sym_info.bid + sym_info.ask) / 2.0 if sym_info else 0.0
                            if rtd_spot > 0:
                                rtd_result = await analyze_options(
                                    rtd_spot, "BOVA11",
                                    win_mapper=win_mapper,
                                    win_symbol=win_symbol,
                                    mt5_conn=mt5_conn,
                                )
                                if rtd_result is not None:
                                    old_cw, old_pw, old_gf = call_wall, put_wall, gamma_flip
                                    call_wall = rtd_result['call_wall']
                                    put_wall = rtd_result['put_wall']
                                    gamma_flip = rtd_result['gamma_flip']

                                    support_zones = rtd_result.get('support_zones', pd.DataFrame())
                                    resist_zones = rtd_result.get('resist_zones', pd.DataFrame())
                                    pin_candidates = rtd_result.get('pin_candidates', pd.DataFrame())
                                    entry_buy_bova, entry_sell_bova = nearest_support_resistance(
                                        rtd_spot, support_zones, resist_zones,
                                        put_wall=put_wall, call_wall=call_wall, pin_candidates=pin_candidates,
                                    )
                                    entry_buy_bova = apply_proximity_offset(entry_buy_bova, 'buy')
                                    entry_sell_bova = apply_proximity_offset(entry_sell_bova, 'sell')
                                    win_entry_buy = win_mapper.bova11_to_ind(entry_buy_bova) if np.isfinite(entry_buy_bova) else np.nan
                                    win_entry_sell = win_mapper.bova11_to_ind(entry_sell_bova) if np.isfinite(entry_sell_bova) else np.nan

                                    print(f"[{ts_now}] [RTD] GEX REFRESHED")
                                    print(f"  Call Wall : {old_cw:.2f} → {call_wall:.2f}" if np.isfinite(call_wall) else f"  Call Wall : N/A")
                                    print(f"  Put Wall  : {old_pw:.2f} → {put_wall:.2f}" if np.isfinite(put_wall) else f"  Put Wall  : N/A")
                                    print(f"  Gamma Flip: {old_gf:.2f} → {gamma_flip:.2f}" if np.isfinite(gamma_flip) else f"  Gamma Flip: N/A")
                                else:
                                    print(f"[{ts_now}] [RTD] Reanalysis returned nothing — keeping old levels")
                        except Exception as e:
                            ts_now = _dt.now().strftime("%H:%M:%S")
                            print(f"[{ts_now}] [RTD] Refresh failed: {e} — keeping old levels")

            # Stop when both sides executed and no open GEX positions remain
            if buy_executed and sell_executed:
                open_gex = _mt5.positions_get(symbol=win_symbol)
                if not any(p.magic == GEX_MAGIC_NUMBER for p in (open_gex or [])):
                    break

            # Before trading window: print waiting message and skip tick logging
            _now_hm_chk = _dt.now().strftime("%H:%M")
            if _now_hm_chk < GEX_TRADE_WINDOW_START:
                if not _waiting_msg_printed:
                    _waiting_msg_printed = True
                    print(f"[{_dt.now().strftime('%H:%M:%S')}] Waiting for trading window "
                          f"({GEX_TRADE_WINDOW_START}) to begin...")
                    if _pre_trade_time > _now_hm_chk:
                        print(f"  Pre-trade GEX refresh scheduled at {_pre_trade_time}")
                await asyncio.sleep(GEX_MONITOR_INTERVAL)
                continue

            tick = _mt5.symbol_info_tick(win_symbol)
            if tick is None:
                print(f"[GEX Monitor] Could not get tick for {win_symbol}")
                await asyncio.sleep(GEX_MONITOR_INTERVAL)
                continue

            win_spot = (tick.bid + tick.ask) / 2.0
            bova_spot = win_mapper.ind_to_bova11(win_spot)

            signal = generate_gex_trade_signals(bova_spot, gamma_flip, call_wall, put_wall,
                                                  support_zones=support_zones if not support_zones.empty else None,
                                                  resist_zones=resist_zones if not resist_zones.empty else None)
            sig = signal['signal']
            strength = signal['strength']
            tick_count += 1

            buy_candidate = (sig == 'BUY' and strength >= GEX_MIN_SIGNAL_STRENGTH)
            sell_candidate = (sig == 'SELL' and strength >= GEX_MIN_SIGNAL_STRENGTH)

            buy_confirm_ticks = buy_confirm_ticks + 1 if buy_candidate else 0
            sell_confirm_ticks = sell_confirm_ticks + 1 if sell_candidate else 0

            buy_confirm_ok = (not GEX_REQUIRE_5M_CONFIRMATION) or (buy_confirm_ticks >= _confirm_ticks)
            sell_confirm_ok = (not GEX_REQUIRE_5M_CONFIRMATION) or (sell_confirm_ticks >= _confirm_ticks)
            neutral_ok = (not GEX_NEUTRAL_ONLY) or is_neutral_setup(
                bova_spot, gamma_flip, call_wall, put_wall, GEX_NEUTRAL_MAX_FLIP_DISTANCE_PCT
            )

            ts = _dt.now().strftime("%H:%M:%S")
            side_status = (("BUY:DONE" if buy_executed else "BUY:wait") + " | "
                           + ("SELL:DONE" if sell_executed else "SELL:wait"))
            print(f"[{ts}] {win_symbol} {win_spot:.0f} | BOVA {bova_spot:.2f} | "
                  f"{sig} [{strength}/3] | {signal['regime']} | {side_status}")

            # Log next DCA threshold for each active side
            acct = _mt5.account_info()
            margin_budget = acct.margin_free * GEX_MARGIN_FREE_PCT if acct else 0.0
            _dca_step = margin_budget * GEX_DCA_LOSS_STEP_PCT
            _pos_snap = _mt5.positions_get(symbol=win_symbol)
            _gex_snap = [p for p in (_pos_snap or []) if p.magic == GEX_MAGIC_NUMBER]
            for _side_label, _trail in [('BUY', trail_buy), ('SELL', trail_sell)]:
                if _trail is None or _trail['active']:
                    continue
                if _trail['dca_count'] >= GEX_DCA_MAX_ORDERS:
                    continue
                _tk_sz = _trail['tick_sz']
                _tk_val = _trail['tick_val']
                _is_buy = (_side_label == 'BUY')
                _side_type = _mt5.POSITION_TYPE_BUY if _is_buy else _mt5.POSITION_TYPE_SELL
                _same = [p for p in _gex_snap if p.type == _side_type]
                _tvol = sum(p.volume for p in _same)
                _pnl_pt = _tvol * (_tk_val / _tk_sz) if _tk_sz > 0 else 0
                if _pnl_pt > 0:
                    _next_level = (_trail['dca_count'] + 1) * _dca_step
                    if _is_buy:
                        _cur_loss_pts = _trail['entry'] - tick.bid
                        _dca_price = _trail['entry'] - (_next_level / _pnl_pt)
                    else:
                        _cur_loss_pts = tick.ask - _trail['entry']
                        _dca_price = _trail['entry'] + (_next_level / _pnl_pt)
                    _dca_price = round(round(_dca_price / _tk_sz) * _tk_sz, 0)
                    _cur_loss_r = _cur_loss_pts * _pnl_pt
                    _remaining = _next_level - _cur_loss_r
                    print(f"       [DCA] {_side_label} #{_trail['dca_count']+1}/{GEX_DCA_MAX_ORDERS} "
                          f"next @ {_dca_price:.0f} (R${_next_level:,.2f} loss) | "
                          f"current R${_cur_loss_r:,.2f} | R${max(_remaining, 0):,.2f} to go")

            # --- Time window & daily loss gate for new entries ---
            _now_hm = _dt.now().strftime("%H:%M")
            _in_trade_window = GEX_TRADE_WINDOW_START <= _now_hm <= GEX_TRADE_WINDOW_END
            _daily_loss_ok = _daily_realized_pnl > -_daily_loss_limit

            # --- Execute BUY ---
            _fib = [1, 1, 2, 3, 5, 8, 13, 21]
            _fib_total = 1 + sum(_fib[:GEX_DCA_MAX_ORDERS])

            if (sig == 'BUY' and strength >= GEX_MIN_SIGNAL_STRENGTH
                    and not buy_executed and np.isfinite(win_entry_buy)
                    and _in_trade_window and _daily_loss_ok
                    and buy_confirm_ok and neutral_ok
                    and _time.time() >= _buy_retry_after):
                _live = _mt5.positions_get(symbol=win_symbol)
                if any(p.magic == GEX_MAGIC_NUMBER and p.type == _mt5.POSITION_TYPE_BUY
                       for p in (_live or [])):
                    buy_executed = True
                    print(f"[GEX Monitor] BUY position already open for magic {GEX_MAGIC_NUMBER} — skipped")
                if not buy_executed:
                    margin_1 = _mt5.order_calc_margin(
                        _mt5.ORDER_TYPE_BUY, win_symbol, 1.0, tick.ask)
                    if margin_1 is not None and margin_1 > 0 and margin_budget > 0:
                        vol = int(margin_budget / (margin_1 * _fib_total))
                        vol = max(vol, int(GEX_ORDER_VOLUME))
                        total_margin = margin_1 * vol * _fib_total
                    else:
                        vol = int(GEX_ORDER_VOLUME)
                        total_margin = margin_1 if margin_1 else 0.0
                    sym_info = _mt5.symbol_info(win_symbol)
                    tick_val = sym_info.trade_tick_value if sym_info else 1.0
                    tick_sz  = sym_info.trade_tick_size if sym_info else 5.0
                    risk_r = margin_budget * GEX_SL_RISK_PCT
                    sl_points = (risk_r / vol) / (tick_val / tick_sz) if vol > 0 and tick_val > 0 else 0.0
                    sl_points = round(sl_points / tick_sz) * tick_sz
                    sl_points = max(sl_points, GEX_MIN_SL_POINTS)
                    sl_price = round(round((tick.ask - sl_points) / tick_sz) * tick_sz, 0) if sl_points > 0 else 0.0
                    tp_price = round(round(win_tp_buy / tick_sz) * tick_sz, 0) if GEX_TP_AT_OPPOSITE_WALL and np.isfinite(win_tp_buy) else 0.0
                    print(f"\n[GEX Monitor] *** BUY TRIGGERED @ {tick.ask:.0f} | {vol} contracts "
                          f"(budget R${margin_budget:,.2f} [{GEX_MARGIN_FREE_PCT:.1%} free], margin R${total_margin:,.2f}, "
                          f"SL {sl_price:.0f}, TP {tp_price:.0f}, risk R${risk_r:,.2f}) ***")
                    result = mt5_conn.place_order(
                        symbol=win_symbol,
                        order_type=_mt5.ORDER_TYPE_BUY,
                        volume=float(vol),
                        price=tick.ask,
                        deviation=GEX_ORDER_DEVIATION,
                        comment=f"GEX BUY {vol}x PutWall {put_wall:.2f}",
                        magic=GEX_MAGIC_NUMBER,
                        sl=sl_price,
                        tp=tp_price,
                    )
                    _entry_budget = margin_budget
                    if result is not None and hasattr(result, 'retcode') and result.retcode == _mt5.TRADE_RETCODE_DONE:
                        buy_executed = True
                        _buy_retry_after = 0.0
                        trail_buy = {
                            'entry': tick.ask, 'vol': vol,
                            'best': tick.ask, 'active': False,
                            'tick_sz': tick_sz, 'tick_val': tick_val,
                            'sl_points': sl_points, 'sl_price': sl_price,
                            'dca_count': 0, 'wall': 'PutWall',
                            'margin_budget': _entry_budget,
                        }
                        _trail_activation = _entry_budget * GEX_TRAILING_ACTIVATION_PCT
                        print(f"[GEX Monitor] BUY FILLED — {vol} contracts, order #{result.order}")
                        print(f"       [Trailing] activates at R${_trail_activation:,.2f} profit "
                              f"({GEX_TRAILING_ACTIVATION_PCT:.0%} of R${_entry_budget:,.2f} budget)")
                        _dca1_step = margin_budget * GEX_DCA_LOSS_STEP_PCT
                        _pnl_pt = vol * (tick_val / tick_sz) if tick_sz > 0 else 0
                        if _pnl_pt > 0:
                            _dca1_price = tick.ask - (_dca1_step / _pnl_pt)
                            _dca1_price = round(round(_dca1_price / tick_sz) * tick_sz, 0)
                            print(f"       [DCA] 1st DCA triggers at {_dca1_price:.0f} "
                                  f"(R${_dca1_step:,.2f} loss, {GEX_DCA_LOSS_STEP_PCT:.0%} of budget)")
                        print()
                    else:
                        # Order did NOT fill. Do NOT mark buy_executed so the
                        # next tick can retry while the BUY signal is still up.
                        # Apply a short cooldown to avoid spamming the broker.
                        _rc = getattr(result, 'retcode', None) if result is not None else None
                        _cm = getattr(result, 'comment', '') if result is not None else _mt5.last_error()
                        _buy_retry_after = _time.time() + _ORDER_RETRY_COOLDOWN
                        print(f"[GEX Monitor] BUY order FAILED (retcode={_rc}, {_cm}) — "
                              f"will retry in {_ORDER_RETRY_COOLDOWN}s if signal persists")
                        print()

            # --- Execute SELL ---
            elif (sig == 'SELL' and strength >= GEX_MIN_SIGNAL_STRENGTH
                      and not sell_executed and np.isfinite(win_entry_sell)
                      and _in_trade_window and _daily_loss_ok
                      and sell_confirm_ok and neutral_ok
                      and _time.time() >= _sell_retry_after):
                _live = _mt5.positions_get(symbol=win_symbol)
                if any(p.magic == GEX_MAGIC_NUMBER and p.type == _mt5.POSITION_TYPE_SELL
                       for p in (_live or [])):
                    sell_executed = True
                    print(f"[GEX Monitor] SELL position already open for magic {GEX_MAGIC_NUMBER} — skipped")
                if not sell_executed:
                    margin_1 = _mt5.order_calc_margin(
                        _mt5.ORDER_TYPE_SELL, win_symbol, 1.0, tick.bid)
                    if margin_1 is not None and margin_1 > 0 and margin_budget > 0:
                        vol = int(margin_budget / (margin_1 * _fib_total))
                        vol = max(vol, int(GEX_ORDER_VOLUME))
                        total_margin = margin_1 * vol * _fib_total
                    else:
                        vol = int(GEX_ORDER_VOLUME)
                        total_margin = margin_1 if margin_1 else 0.0
                    sym_info = _mt5.symbol_info(win_symbol)
                    tick_val = sym_info.trade_tick_value if sym_info else 1.0
                    tick_sz  = sym_info.trade_tick_size if sym_info else 5.0
                    risk_r = margin_budget * GEX_SL_RISK_PCT
                    sl_points = (risk_r / vol) / (tick_val / tick_sz) if vol > 0 and tick_val > 0 else 0.0
                    sl_points = round(sl_points / tick_sz) * tick_sz
                    sl_points = max(sl_points, GEX_MIN_SL_POINTS)
                    sl_price = round(round((tick.bid + sl_points) / tick_sz) * tick_sz, 0) if sl_points > 0 else 0.0
                    tp_price = round(round(win_tp_sell / tick_sz) * tick_sz, 0) if GEX_TP_AT_OPPOSITE_WALL and np.isfinite(win_tp_sell) else 0.0
                    print(f"\n[GEX Monitor] *** SELL TRIGGERED @ {tick.bid:.0f} | {vol} contracts "
                          f"(budget R${margin_budget:,.2f} [{GEX_MARGIN_FREE_PCT:.1%} free], margin R${total_margin:,.2f}, "
                          f"SL {sl_price:.0f}, TP {tp_price:.0f}, risk R${risk_r:,.2f}) ***")
                    result = mt5_conn.place_order(
                        symbol=win_symbol,
                        order_type=_mt5.ORDER_TYPE_SELL,
                        volume=float(vol),
                        price=tick.bid,
                        deviation=GEX_ORDER_DEVIATION,
                        comment=f"GEX SELL {vol}x CallWall {call_wall:.2f}",
                        magic=GEX_MAGIC_NUMBER,
                        sl=sl_price,
                        tp=tp_price,
                    )
                    _entry_budget = margin_budget
                    if result is not None and hasattr(result, 'retcode') and result.retcode == _mt5.TRADE_RETCODE_DONE:
                        sell_executed = True
                        _sell_retry_after = 0.0
                        trail_sell = {
                            'entry': tick.bid, 'vol': vol,
                            'best': tick.bid, 'active': False,
                            'tick_sz': tick_sz, 'tick_val': tick_val,
                            'sl_points': sl_points, 'sl_price': sl_price,
                            'dca_count': 0, 'wall': 'CallWall',
                            'margin_budget': _entry_budget,
                        }
                        _trail_activation = _entry_budget * GEX_TRAILING_ACTIVATION_PCT
                        print(f"[GEX Monitor] SELL FILLED — {vol} contracts, order #{result.order}")
                        print(f"       [Trailing] activates at R${_trail_activation:,.2f} profit "
                              f"({GEX_TRAILING_ACTIVATION_PCT:.0%} of R${_entry_budget:,.2f} budget)")
                        _dca1_step = margin_budget * GEX_DCA_LOSS_STEP_PCT
                        _pnl_pt = vol * (tick_val / tick_sz) if tick_sz > 0 else 0
                        if _pnl_pt > 0:
                            _dca1_price = tick.bid + (_dca1_step / _pnl_pt)
                            _dca1_price = round(round(_dca1_price / tick_sz) * tick_sz, 0)
                            print(f"       [DCA] 1st DCA triggers at {_dca1_price:.0f} "
                                  f"(R${_dca1_step:,.2f} loss, {GEX_DCA_LOSS_STEP_PCT:.0%} of budget)")
                        print()
                    else:
                        # Order did NOT fill. Keep sell_executed False so the
                        # next tick retries while the SELL signal is still up.
                        _rc = getattr(result, 'retcode', None) if result is not None else None
                        _cm = getattr(result, 'comment', '') if result is not None else _mt5.last_error()
                        _sell_retry_after = _time.time() + _ORDER_RETRY_COOLDOWN
                        print(f"[GEX Monitor] SELL order FAILED (retcode={_rc}, {_cm}) — "
                              f"will retry in {_ORDER_RETRY_COOLDOWN}s if signal persists")
                        print()

            # ---- Trailing stop management ----
            positions = _mt5.positions_get(symbol=win_symbol)
            gex_positions = [p for p in (positions or []) if p.magic == GEX_MAGIC_NUMBER]

            _sides_processed = set()
            for pos in gex_positions:
                is_buy = (pos.type == _mt5.POSITION_TYPE_BUY)
                side_key = 'BUY' if is_buy else 'SELL'
                if side_key in _sides_processed:
                    continue
                _sides_processed.add(side_key)

                trail = trail_buy if is_buy else trail_sell
                if trail is None:
                    continue

                _trail_budget = trail.get('margin_budget', margin_budget)
                activation_r = _trail_budget * GEX_TRAILING_ACTIVATION_PCT

                tk_sz = trail['tick_sz']
                tk_val = trail['tick_val']

                side_type = _mt5.POSITION_TYPE_BUY if is_buy else _mt5.POSITION_TYPE_SELL
                same_side = [p for p in gex_positions if p.type == side_type]
                total_vol = sum(p.volume for p in same_side)
                pnl_per_pt = total_vol * (tk_val / tk_sz)

                if is_buy:
                    trail['best'] = max(trail['best'], tick.bid)
                    profit_pts = trail['best'] - trail['entry']
                else:
                    trail['best'] = min(trail['best'], tick.ask)
                    profit_pts = trail['entry'] - trail['best']

                profit_r = profit_pts * pnl_per_pt

                _pct_to_activation = (profit_r / activation_r * 100) if activation_r > 0 else 0
                if not trail['active']:
                    print(f"       [Trail] {side_key} profit R${profit_r:,.2f} / "
                          f"R${activation_r:,.2f} ({_pct_to_activation:.0f}%) "
                          f"| best {trail['best']:.0f}")

                if not trail['active'] and profit_r >= activation_r:
                    trail['active'] = True
                    ts_now = _dt.now().strftime("%H:%M:%S")
                    print(f"[{ts_now}] [Trailing] {side_key} activated — "
                          f"profit R${profit_r:,.2f} >= R${activation_r:,.2f} "
                          f"({len(same_side)} positions, {total_vol:.0f} contracts)")

                if trail['active']:
                    trail_dist = trail['sl_points'] * GEX_TRAILING_DISTANCE_FACTOR
                    trail_dist = max(trail_dist, GEX_MIN_SL_POINTS)
                    if is_buy:
                        new_sl = trail['best'] - trail_dist
                        new_sl = round(round(new_sl / tk_sz) * tk_sz, 0)
                    else:
                        new_sl = trail['best'] + trail_dist
                        new_sl = round(round(new_sl / tk_sz) * tk_sz, 0)

                    for sp in same_side:
                        current_sl = sp.sl
                        should_update = False
                        if is_buy and (current_sl == 0.0 or new_sl > current_sl):
                            should_update = True
                        elif not is_buy and (current_sl == 0.0 or new_sl < current_sl):
                            should_update = True

                        if should_update:
                            safe_sl = sanitize_modify_sl(
                                win_symbol, is_buy, new_sl, current_sl=current_sl,
                            )
                            if safe_sl <= 0.0:
                                continue  # broker would reject; skip this tick
                            modify_req = {
                                "action": _mt5.TRADE_ACTION_SLTP,
                                "symbol": win_symbol,
                                "position": sp.ticket,
                                "sl": safe_sl,
                                "tp": sp.tp,
                            }
                            mod_result = _mt5.order_send(modify_req)
                            ts_now = _dt.now().strftime("%H:%M:%S")
                            if mod_result and mod_result.retcode == _mt5.TRADE_RETCODE_DONE:
                                print(f"[{ts_now}] [Trailing] {side_key} #{sp.ticket} SL → "
                                      f"{safe_sl:.0f} (best {trail['best']:.0f}, "
                                      f"profit R${profit_r:,.2f})")
                            else:
                                err = mod_result.comment if mod_result else _mt5.last_error()
                                print(f"[{ts_now}] [Trailing] SL update #{sp.ticket} failed: {err} "
                                      f"(tried {safe_sl:.0f})")

                    trail['sl_price'] = new_sl

            # ---- DCA: add orders on losing positions at each 10% total margin step ----
            dca_step_r = margin_budget * GEX_DCA_LOSS_STEP_PCT

            _dca_sides_processed = set()
            for pos in gex_positions:
                is_buy = (pos.type == _mt5.POSITION_TYPE_BUY)
                side_key = 'BUY' if is_buy else 'SELL'
                if side_key in _dca_sides_processed:
                    continue
                _dca_sides_processed.add(side_key)

                trail = trail_buy if is_buy else trail_sell
                if trail is None or trail['active']:
                    continue
                if trail['dca_count'] >= GEX_DCA_MAX_ORDERS:
                    continue
                if '_dca_cooldown' in trail and (_dt.now() - trail['_dca_cooldown']).total_seconds() < 60:
                    continue

                tk_sz = trail['tick_sz']
                tk_val = trail['tick_val']

                side_type = _mt5.POSITION_TYPE_BUY if is_buy else _mt5.POSITION_TYPE_SELL
                same_side = [p for p in gex_positions if p.type == side_type]
                total_vol = sum(p.volume for p in same_side)
                pnl_per_pt = total_vol * (tk_val / tk_sz)

                if is_buy:
                    loss_pts = trail['entry'] - tick.bid
                else:
                    loss_pts = tick.ask - trail['entry']
                loss_r = loss_pts * pnl_per_pt

                levels_crossed = int(loss_r / dca_step_r) if dca_step_r > 0 else 0
                needed = levels_crossed - trail['dca_count']

                while needed > 0:
                    if trail['dca_count'] >= GEX_DCA_MAX_ORDERS:
                        break
                    fib_idx = trail['dca_count'] if trail['dca_count'] < len(_fib) else len(_fib) - 1
                    dca_vol = trail['vol'] * _fib[fib_idx]

                    margin_1_dca = _mt5.order_calc_margin(
                        _mt5.ORDER_TYPE_BUY if is_buy else _mt5.ORDER_TYPE_SELL,
                        win_symbol, float(dca_vol),
                        tick.ask if is_buy else tick.bid)
                    current_margin_used = sum(p.volume for p in same_side) * (margin_1_dca / dca_vol if dca_vol > 0 and margin_1_dca else 0)
                    if margin_1_dca is not None and margin_budget > 0:
                        if current_margin_used + margin_1_dca > margin_budget:
                            ts_now = _dt.now().strftime("%H:%M:%S")
                            print(f"[{ts_now}] [DCA] SKIPPED — margin would exceed budget "
                                  f"(used R${current_margin_used:,.2f} + R${margin_1_dca:,.2f} > budget R${margin_budget:,.2f})")
                            break

                    if is_buy:
                        dca_price = tick.ask
                        dca_type = _mt5.ORDER_TYPE_BUY
                    else:
                        dca_price = tick.bid
                        dca_type = _mt5.ORDER_TYPE_SELL

                    new_total_vol_proj = total_vol + dca_vol
                    new_avg_entry = (trail['entry'] * total_vol + dca_price * dca_vol) / new_total_vol_proj
                    pnl_per_pt_new = new_total_vol_proj * (tk_val / tk_sz)
                    risk_r = margin_budget * GEX_SL_RISK_PCT
                    new_sl_dist = (risk_r / pnl_per_pt_new) if pnl_per_pt_new > 0 else 0.0
                    new_sl_dist = round(new_sl_dist / tk_sz) * tk_sz
                    new_sl_dist = max(new_sl_dist, GEX_MIN_SL_POINTS)

                    actual_risk = new_sl_dist * pnl_per_pt_new
                    if actual_risk > risk_r * 2:
                        ts_now = _dt.now().strftime("%H:%M:%S")
                        print(f"[{ts_now}] [DCA] SKIPPED — min SL floor would expose "
                              f"R${actual_risk:,.2f} risk (> 2× budget R${risk_r:,.2f})")
                        break

                    if is_buy:
                        new_sl = round(round((new_avg_entry - new_sl_dist) / tk_sz) * tk_sz, 0)
                    else:
                        new_sl = round(round((new_avg_entry + new_sl_dist) / tk_sz) * tk_sz, 0)

                    dca_n = trail['dca_count'] + 1
                    ts_now = _dt.now().strftime("%H:%M:%S")
                    dca_tp = same_side[0].tp if same_side else 0.0
                    print(f"\n[{ts_now}] [DCA] {'BUY' if is_buy else 'SELL'} #{dca_n}/{GEX_DCA_MAX_ORDERS} "
                          f"@ {dca_price:.0f} | loss R${loss_r:,.2f} | +{dca_vol} contracts "
                          f"| new SL {new_sl:.0f} (risk capped R${risk_r:,.2f})")
                    wall_label = trail.get('wall', 'Wall')
                    dca_result = mt5_conn.place_order(
                        symbol=win_symbol,
                        order_type=dca_type,
                        volume=float(dca_vol),
                        price=dca_price,
                        deviation=GEX_ORDER_DEVIATION,
                        comment=f"GEX DCA#{dca_n} {wall_label}",
                        magic=GEX_MAGIC_NUMBER,
                        sl=new_sl,
                        tp=dca_tp,
                    )
                    if dca_result is not None and hasattr(dca_result, 'retcode') and dca_result.retcode == _mt5.TRADE_RETCODE_DONE:
                        trail['dca_count'] += 1
                        new_total_vol = total_vol + dca_vol
                        trail['entry'] = (trail['entry'] * total_vol + dca_price * dca_vol) / new_total_vol
                        trail['sl_points'] = new_sl_dist
                        trail['sl_price'] = new_sl
                        total_vol = new_total_vol
                        print(f"[{ts_now}] [DCA] FILLED — avg entry → {trail['entry']:.0f}, "
                              f"total {new_total_vol:.0f} contracts, SL → {new_sl:.0f}")

                        for sp in same_side:
                            current_sp_sl = sp.sl
                            should_update = False
                            if is_buy and (current_sp_sl == 0.0 or new_sl != current_sp_sl):
                                should_update = True
                            elif not is_buy and (current_sp_sl == 0.0 or new_sl != current_sp_sl):
                                should_update = True
                            if should_update:
                                safe_sl = sanitize_modify_sl(
                                    win_symbol, is_buy, new_sl, current_sl=current_sp_sl,
                                )
                                if safe_sl <= 0.0:
                                    continue
                                modify_req = {
                                    "action": _mt5.TRADE_ACTION_SLTP,
                                    "symbol": win_symbol,
                                    "position": sp.ticket,
                                    "sl": safe_sl,
                                    "tp": sp.tp,
                                }
                                mod_result = _mt5.order_send(modify_req)
                                if mod_result and mod_result.retcode == _mt5.TRADE_RETCODE_DONE:
                                    print(f"[{ts_now}] [DCA] SL synced #{sp.ticket} → {safe_sl:.0f}")
                                else:
                                    err = mod_result.comment if mod_result else _mt5.last_error()
                                    print(f"[{ts_now}] [DCA] SL sync #{sp.ticket} failed: {err} "
                                          f"(tried {safe_sl:.0f})")

                        pnl_per_pt = total_vol * (tk_val / tk_sz)
                        if is_buy:
                            loss_pts = trail['entry'] - tick.bid
                        else:
                            loss_pts = tick.ask - trail['entry']
                        loss_r = loss_pts * pnl_per_pt
                        levels_crossed = int(loss_r / dca_step_r) if dca_step_r > 0 else 0
                        needed = levels_crossed - trail['dca_count']
                    else:
                        trail['_dca_cooldown'] = _dt.now()
                        print(f"[{ts_now}] [DCA] Order FAILED — cooldown 60s (check logs above)")
                        break

            # --- Track daily realized P&L from closed GEX positions ---
            _today_start = _dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
            _deals = _mt5.history_deals_get(_today_start, _dt.now(), group=f"*{win_symbol}*")
            _daily_realized_pnl = sum(
                d.profit for d in (_deals or [])
                if d.magic == GEX_MAGIC_NUMBER and d.entry == _mt5.DEAL_ENTRY_OUT
            )
            if _daily_realized_pnl <= -_daily_loss_limit:
                ts_now = _dt.now().strftime("%H:%M:%S")
                print(f"[{ts_now}] [RISK] Daily loss R${_daily_realized_pnl:,.2f} "
                      f">= limit R${-_daily_loss_limit:,.2f} — new entries halted")

            await asyncio.sleep(GEX_MONITOR_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n[GEX Monitor] Stopped by user after {tick_count} ticks.")

    print(f"[GEX Monitor] Session ended. BUY={'DONE' if buy_executed else 'PENDING'} | "
          f"SELL={'DONE' if sell_executed else 'PENDING'}")

# -*- coding: utf-8 -*-
"""
DCA GEX Strategy Simulation for WIN (Mini Ibovespa Futures)
===========================================================
Simulates price going DOWN -5% and UP +5% from entry, applying the
exact DCA logic from main.py with constants.py parameters.

WIN contract specs (B3):
    tick_size  = 5 pts
    tick_value = R$0.20 per point per contract  (1 tick = R$1.00)
"""

import numpy as np
from constants import (
    GEX_MARGIN_FREE_PCT,
    GEX_SL_RISK_PCT,
    GEX_TRAILING_ACTIVATION_PCT,
    GEX_DCA_LOSS_STEP_PCT,
    GEX_DCA_MAX_ORDERS,
    GEX_ORDER_VOLUME,
    GEX_WALL_PROXIMITY_PCT,
    GEX_MIN_SIGNAL_STRENGTH,
)

# ── WIN Contract Specifications ──────────────────────────────────────
TICK_SIZE  = 5       # points
TICK_VALUE = 0.20    # R$ per point per contract (tick_val/tick_sz = 0.20)
PNL_PER_POINT = TICK_VALUE  # R$0.20 per point per contract

# ── Simulation Assumptions ───────────────────────────────────────────
WIN_SPOT        = 130_000       # Current WIN price (Ibovespa index points)
FREE_MARGIN     = 10_000.0      # R$ free margin available
MARGIN_PER_LOT  = 100.0         # R$ margin required per 1 WIN mini contract
SCENARIO_PCT    = 0.05          # 5% move

# ── Derived from constants.py ────────────────────────────────────────
FIB_SEQ = [1, 1, 2, 3, 5, 8, 13, 21]
FIB_TOTAL = 1 + sum(FIB_SEQ[:GEX_DCA_MAX_ORDERS])  # initial(1x) + DCA sum
MARGIN_BUDGET = FREE_MARGIN * GEX_MARGIN_FREE_PCT

# ── GEX Levels (illustrative, based on typical BOVA->WIN mapping) ───
# Assume BOVA11 ~ 130.00 → WIN ~ 130,000 (beta ~ 1000)
BOVA_SPOT     = 130.00
CALL_WALL     = 133.00   # BOVA call wall
PUT_WALL      = 127.00   # BOVA put wall
GAMMA_FLIP    = 129.50   # BOVA gamma flip

WIN_CALL_WALL  = 133_000
WIN_PUT_WALL   = 127_000
WIN_GAMMA_FLIP = 129_500

ENTRY_BUY_BOVA  = PUT_WALL * (1.0 + GEX_WALL_PROXIMITY_PCT)
ENTRY_SELL_BOVA = CALL_WALL * (1.0 - GEX_WALL_PROXIMITY_PCT)
WIN_ENTRY_BUY   = int(ENTRY_BUY_BOVA * 1000)
WIN_ENTRY_SELL  = int(ENTRY_SELL_BOVA * 1000)


def align_tick(price):
    """Round price to nearest tick."""
    return round(round(price / TICK_SIZE) * TICK_SIZE)


def compute_initial_volume():
    """Compute initial volume sized to fit entire DCA plan within margin budget."""
    if MARGIN_PER_LOT > 0 and MARGIN_BUDGET > 0:
        vol = int(MARGIN_BUDGET / (MARGIN_PER_LOT * FIB_TOTAL))
        vol = max(vol, int(GEX_ORDER_VOLUME))
    else:
        vol = int(GEX_ORDER_VOLUME)
    return vol


def compute_sl(entry_price, vol, is_buy):
    """Compute stop-loss price from risk budget."""
    risk_r = MARGIN_BUDGET * GEX_SL_RISK_PCT
    sl_points = (risk_r / vol) / PNL_PER_POINT if vol > 0 else 0.0
    sl_points = round(sl_points / TICK_SIZE) * TICK_SIZE
    if is_buy:
        return align_tick(entry_price - sl_points), sl_points
    else:
        return align_tick(entry_price + sl_points), sl_points


def simulate_scenario(entry_price, is_buy, target_pct, label):
    """
    Simulate the DCA GEX strategy for a single side (BUY or SELL).

    Parameters
    ----------
    entry_price : float   WIN entry price
    is_buy      : bool    True = BUY (long), False = SELL (short)
    target_pct  : float   Signed price change (e.g., -0.05 or +0.05)
    label       : str     Scenario label
    """
    side = "BUY" if is_buy else "SELL"
    vol = compute_initial_volume()
    sl_price, sl_points = compute_sl(entry_price, vol, is_buy)
    risk_r = MARGIN_BUDGET * GEX_SL_RISK_PCT
    dca_step_r = MARGIN_BUDGET * GEX_DCA_LOSS_STEP_PCT
    activation_r = MARGIN_BUDGET * GEX_TRAILING_ACTIVATION_PCT

    target_price = align_tick(entry_price * (1.0 + target_pct))

    # ── Header ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SCENARIO: {label}")
    print(f"  Side: {side}  |  Entry: {entry_price:,.0f}  |  Target: {target_price:,.0f} ({target_pct:+.1%})")
    print(f"{'='*80}")
    print(f"  Parameters from constants.py:")
    print(f"    GEX_MARGIN_FREE_PCT       = {GEX_MARGIN_FREE_PCT:.1%}")
    print(f"    GEX_SL_RISK_PCT           = {GEX_SL_RISK_PCT:.1%}")
    print(f"    GEX_TRAILING_ACTIVATION   = {GEX_TRAILING_ACTIVATION_PCT:.1%}")
    print(f"    GEX_DCA_LOSS_STEP_PCT     = {GEX_DCA_LOSS_STEP_PCT:.1%}")
    print(f"    GEX_DCA_MAX_ORDERS        = {GEX_DCA_MAX_ORDERS}")
    print(f"    GEX_WALL_PROXIMITY_PCT    = {GEX_WALL_PROXIMITY_PCT}")
    print(f"    Fibonacci DCA Sequence    = {FIB_SEQ[:GEX_DCA_MAX_ORDERS]}")
    print(f"    Fib Total Multiplier      = {FIB_TOTAL}x initial volume")
    print()
    print(f"  Account:")
    print(f"    Free Margin               = R$ {FREE_MARGIN:,.2f}")
    print(f"    Margin Budget (5%)        = R$ {MARGIN_BUDGET:,.2f}")
    print(f"    Margin per Contract       = R$ {MARGIN_PER_LOT:,.2f}")
    print(f"    Initial Volume            = {vol} contracts")
    print(f"    Total Margin (all DCA)    = R$ {MARGIN_PER_LOT * vol * FIB_TOTAL:,.2f}")
    print()
    print(f"  Risk:")
    print(f"    SL Risk Budget (50%)      = R$ {risk_r:,.2f}")
    print(f"    SL Distance               = {sl_points:,.0f} pts")
    print(f"    SL Price                  = {sl_price:,.0f}")
    print(f"    DCA Loss Step (10%)       = R$ {dca_step_r:,.2f}")
    print(f"    Trailing Activation (35%) = R$ {activation_r:,.2f}")
    print()

    # ── Simulate price movement in 100 steps ─────────────────────────
    n_steps = 200
    prices = np.linspace(entry_price, target_price, n_steps)

    # State
    positions = [{'entry': entry_price, 'vol': vol, 'label': 'Initial'}]
    avg_entry = entry_price
    total_vol = vol
    dca_count = 0
    best_price = entry_price
    trailing_active = False
    current_sl = sl_price
    stopped_out = False
    stopped_price = None
    trailing_triggered_at = None

    # Track DCA events and trailing events
    events = []

    for step, current_price in enumerate(prices):
        price = align_tick(current_price)

        # Update best price for trailing
        if is_buy:
            best_price = max(best_price, price)
        else:
            best_price = min(best_price, price)

        # P&L calculation
        if is_buy:
            pnl_pts = price - avg_entry
        else:
            pnl_pts = avg_entry - price

        pnl_r = pnl_pts * total_vol * PNL_PER_POINT
        pnl_per_pt_total = total_vol * PNL_PER_POINT

        # ── Check stop-loss ──────────────────────────────────
        if is_buy and price <= current_sl:
            stopped_out = True
            stopped_price = current_sl
            pnl_pts = current_sl - avg_entry
            pnl_r = pnl_pts * total_vol * PNL_PER_POINT
            events.append(('STOP-LOSS', step, current_sl, pnl_r, total_vol, dca_count))
            break
        elif not is_buy and price >= current_sl:
            stopped_out = True
            stopped_price = current_sl
            pnl_pts = avg_entry - current_sl
            pnl_r = pnl_pts * total_vol * PNL_PER_POINT
            events.append(('STOP-LOSS', step, current_sl, pnl_r, total_vol, dca_count))
            break

        # ── DCA logic (only when losing, before trailing activates) ──
        if not trailing_active and dca_count < GEX_DCA_MAX_ORDERS:
            if is_buy:
                loss_pts = avg_entry - price  # positive when losing for BUY
            else:
                loss_pts = price - avg_entry  # positive when losing for SELL

            loss_r = loss_pts * total_vol * PNL_PER_POINT

            if loss_r > 0:  # actually losing
                levels_crossed = int(loss_r / dca_step_r) if dca_step_r > 0 else 0
                needed = levels_crossed - dca_count

                for _ in range(needed):
                    if dca_count >= GEX_DCA_MAX_ORDERS:
                        break
                    fib_idx = min(dca_count, len(FIB_SEQ) - 1)
                    dca_vol = vol * FIB_SEQ[fib_idx]

                    # --- Cap DCA: skip if margin would exceed budget ---
                    margin_used = total_vol * MARGIN_PER_LOT
                    margin_needed = dca_vol * MARGIN_PER_LOT
                    if margin_used + margin_needed > MARGIN_BUDGET:
                        events.append(('DCA_SKIPPED', step, price, loss_r, total_vol, dca_count))
                        break

                    # Update average entry
                    new_total = total_vol + dca_vol
                    new_avg = (avg_entry * total_vol + price * dca_vol) / new_total

                    # --- Recalculate SL to bound aggregate risk within budget ---
                    pnl_per_pt_new = new_total * PNL_PER_POINT
                    new_sl_dist = (risk_r / pnl_per_pt_new) if pnl_per_pt_new > 0 else 0.0
                    new_sl_dist = round(new_sl_dist / TICK_SIZE) * TICK_SIZE
                    if is_buy:
                        current_sl = align_tick(new_avg - new_sl_dist)
                    else:
                        current_sl = align_tick(new_avg + new_sl_dist)
                    sl_points = new_sl_dist  # update for trailing

                    avg_entry = new_avg
                    total_vol = new_total
                    dca_count += 1

                    # Recalc P&L with new avg
                    if is_buy:
                        new_pnl_pts = price - avg_entry
                    else:
                        new_pnl_pts = avg_entry - price
                    new_pnl_r = new_pnl_pts * total_vol * PNL_PER_POINT

                    positions.append({
                        'entry': price,
                        'vol': dca_vol,
                        'label': f'DCA #{dca_count} (Fib[{fib_idx}]={FIB_SEQ[fib_idx]}x)'
                    })
                    events.append(('DCA', step, price, new_pnl_r, total_vol, dca_count))

        # ── Trailing stop activation ─────────────────────────
        if is_buy:
            profit_pts = best_price - avg_entry
        else:
            profit_pts = avg_entry - best_price

        profit_r = profit_pts * total_vol * PNL_PER_POINT

        if not trailing_active and profit_r >= activation_r:
            trailing_active = True
            trailing_triggered_at = price
            events.append(('TRAIL_ACTIVATED', step, price, profit_r, total_vol, dca_count))

        if trailing_active:
            if is_buy:
                new_sl = align_tick(best_price - sl_points)
                if new_sl > current_sl:
                    current_sl = new_sl
            else:
                new_sl = align_tick(best_price + sl_points)
                if new_sl < current_sl:
                    current_sl = new_sl

    # ── Final state ──────────────────────────────────────────────────
    final_price = stopped_price if stopped_out else align_tick(target_price)
    if is_buy:
        final_pnl_pts = final_price - avg_entry
    else:
        final_pnl_pts = avg_entry - final_price
    final_pnl_r = final_pnl_pts * total_vol * PNL_PER_POINT

    # ── Print Position Ladder ────────────────────────────────────────
    print(f"  {'─'*76}")
    print(f"  POSITION LADDER:")
    print(f"  {'─'*76}")
    print(f"  {'#':<4} {'Label':<28} {'Entry':>10} {'Volume':>8} {'Margin':>12}")
    print(f"  {'─'*76}")
    total_margin_used = 0
    for i, pos in enumerate(positions):
        m = pos['vol'] * MARGIN_PER_LOT
        total_margin_used += m
        print(f"  {i+1:<4} {pos['label']:<28} {pos['entry']:>10,.0f} {pos['vol']:>8} R$ {m:>9,.2f}")
    print(f"  {'─'*76}")
    print(f"  {'TOTAL':<33} {'avg:':>2}{avg_entry:>7,.0f} {total_vol:>8} R$ {total_margin_used:>9,.2f}")
    print()

    # ── Print Event Timeline ─────────────────────────────────────────
    print(f"  {'─'*76}")
    print(f"  EVENT TIMELINE:")
    print(f"  {'─'*76}")
    print(f"  {'Step':<6} {'Event':<20} {'Price':>10} {'P&L':>12} {'TotalVol':>10} {'DCA#':>5}")
    print(f"  {'─'*76}")
    for ev_type, ev_step, ev_price, ev_pnl, ev_vol, ev_dca in events:
        pnl_color = '+' if ev_pnl >= 0 else ''
        print(f"  {ev_step:<6} {ev_type:<20} {ev_price:>10,.0f} {pnl_color}R$ {ev_pnl:>9,.2f} {ev_vol:>10} {ev_dca:>5}")
    if not events:
        print(f"  {'(no DCA or trailing events triggered)'}")
    print()

    # ── Final P&L Summary ────────────────────────────────────────────
    print(f"  {'─'*76}")
    print(f"  FINAL RESULT:")
    print(f"  {'─'*76}")
    print(f"    Entry Price (avg)  : {avg_entry:>12,.0f}")
    print(f"    Exit Price         : {final_price:>12,.0f} {'(STOPPED)' if stopped_out else '(TARGET)'}")
    print(f"    Move from Entry    : {final_price - entry_price:>+12,.0f} pts ({(final_price/entry_price - 1):>+.2%})")
    print(f"    Total Contracts    : {total_vol:>12}")
    print(f"    DCA Orders Used    : {dca_count:>12} / {GEX_DCA_MAX_ORDERS}")
    print(f"    P&L (points)       : {final_pnl_pts:>+12,.0f} pts")
    print(f"    P&L (R$)           : R$ {final_pnl_r:>+11,.2f}")
    print(f"    ROI on Budget      : {(final_pnl_r / MARGIN_BUDGET * 100):>+11.1f}%")
    print(f"    Trailing Active    : {'Yes @ ' + f'{trailing_triggered_at:,.0f}' if trailing_active else 'No'}")
    print(f"    Final SL           : {current_sl:>12,.0f}")
    if stopped_out:
        print(f"    >>> POSITION STOPPED OUT AT {stopped_price:,.0f} <<<")
    print(f"  {'─'*76}")

    return {
        'label': label,
        'side': side,
        'entry': entry_price,
        'avg_entry': avg_entry,
        'exit': final_price,
        'total_vol': total_vol,
        'dca_count': dca_count,
        'pnl_r': final_pnl_r,
        'pnl_pts': final_pnl_pts,
        'stopped': stopped_out,
        'trailing_active': trailing_active,
    }


def main():
    print(r"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              DCA GEX STRATEGY — WIN SIMULATION REPORT                  ║
    ║                   Price Scenarios: -5% and +5%                         ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"  GEX Levels (illustrative BOVA11 → WIN mapping):")
    print(f"    WIN Spot       : {WIN_SPOT:>10,}")
    print(f"    Call Wall      : {WIN_CALL_WALL:>10,}  (BOVA {CALL_WALL:.2f})")
    print(f"    Put Wall       : {WIN_PUT_WALL:>10,}  (BOVA {PUT_WALL:.2f})")
    print(f"    Gamma Flip     : {WIN_GAMMA_FLIP:>10,}  (BOVA {GAMMA_FLIP:.2f})")
    print(f"    Entry BUY zone : {WIN_ENTRY_BUY:>10,}  (BOVA {ENTRY_BUY_BOVA:.2f})")
    print(f"    Entry SELL zone: {WIN_ENTRY_SELL:>10,}  (BOVA {ENTRY_SELL_BOVA:.2f})")
    print()

    results = []

    # ── Scenario 1: BUY at Put Wall, price drops 5% ─────────────────
    # Worst case for BUY: entered long, market keeps falling
    r1 = simulate_scenario(
        entry_price=WIN_ENTRY_BUY,
        is_buy=True,
        target_pct=-SCENARIO_PCT,
        label="BUY @ Put Wall → Price DROPS -5%"
    )
    results.append(r1)

    # ── Scenario 2: BUY at Put Wall, price rises 5% ─────────────────
    # Best case for BUY: entered long, market bounces up
    r2 = simulate_scenario(
        entry_price=WIN_ENTRY_BUY,
        is_buy=True,
        target_pct=+SCENARIO_PCT,
        label="BUY @ Put Wall → Price RISES +5%"
    )
    results.append(r2)

    # ── Scenario 3: SELL at Call Wall, price rises 5% ────────────────
    # Worst case for SELL: entered short, market keeps rising
    r3 = simulate_scenario(
        entry_price=WIN_ENTRY_SELL,
        is_buy=False,
        target_pct=+SCENARIO_PCT,
        label="SELL @ Call Wall → Price RISES +5%"
    )
    results.append(r3)

    # ── Scenario 4: SELL at Call Wall, price drops 5% ────────────────
    # Best case for SELL: entered short, market drops
    r4 = simulate_scenario(
        entry_price=WIN_ENTRY_SELL,
        is_buy=False,
        target_pct=-SCENARIO_PCT,
        label="SELL @ Call Wall → Price DROPS -5%"
    )
    results.append(r4)

    # ── Summary Table ────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"  COMPARATIVE SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Scenario':<42} {'Entry':>8} {'Exit':>8} {'Avg':>8} {'Vol':>5} {'DCA':>4} {'P&L R$':>12} {'ROI%':>8} {'Stop':>5} {'Trail':>5}")
    print(f"  {'─'*98}")
    for r in results:
        roi = r['pnl_r'] / MARGIN_BUDGET * 100
        print(f"  {r['label']:<42} {r['entry']:>8,.0f} {r['exit']:>8,.0f} {r['avg_entry']:>8,.0f} "
              f"{r['total_vol']:>5} {r['dca_count']:>4} "
              f"{'R$':>1}{r['pnl_r']:>+10,.2f} {roi:>+7.1f}% "
              f"{'YES' if r['stopped'] else 'no':>5} "
              f"{'YES' if r['trailing_active'] else 'no':>5}")
    print(f"  {'─'*98}")

    # ── Key takeaways ────────────────────────────────────────────────
    print(f"\n  KEY OBSERVATIONS:")
    print(f"  ─────────────────")
    worst = min(results, key=lambda r: r['pnl_r'])
    best = max(results, key=lambda r: r['pnl_r'])
    print(f"    Worst case : {worst['label']}")
    print(f"                 P&L = R$ {worst['pnl_r']:+,.2f} ({worst['pnl_r']/MARGIN_BUDGET*100:+.1f}% of budget)")
    print(f"    Best case  : {best['label']}")
    print(f"                 P&L = R$ {best['pnl_r']:+,.2f} ({best['pnl_r']/MARGIN_BUDGET*100:+.1f}% of budget)")
    print()
    print(f"    Max loss capped by SL at {GEX_SL_RISK_PCT:.0%} of margin budget = R$ {MARGIN_BUDGET * GEX_SL_RISK_PCT:,.2f}")
    print(f"    DCA Fibonacci scaling increases avg position on drawdowns")
    print(f"    Trailing stop activates at {GEX_TRAILING_ACTIVATION_PCT:.0%} profit (R$ {MARGIN_BUDGET * GEX_TRAILING_ACTIVATION_PCT:,.2f})")
    print(f"    Total DCA plan: {FIB_TOTAL}x initial volume across {GEX_DCA_MAX_ORDERS} additions")
    print()


if __name__ == "__main__":
    main()

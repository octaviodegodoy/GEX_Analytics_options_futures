# -*- coding: utf-8 -*-
"""
GEX Plot Functions — all matplotlib charting for GEX analytics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_notional_by_strike(vol_by_strike, spot, underlying, show_plots=False):
    """Stacked bar chart of notional volume by strike."""
    if vol_by_strike.empty or vol_by_strike.select_dtypes(include='number').empty:
        print(f"[!] No notional data to plot for {underlying} — skipping chart.")
        return
    vol_by_strike.plot(kind='bar', stacked=True, figsize=(12, 12),
                       color=['#2563EB', '#EF4444'], alpha=0.7)
    plt.axvline(np.argmin(np.abs(vol_by_strike.index - spot)),
                color='black', linestyle='--')
    plt.title(f"Volume Financeiro por Strike — {underlying}")
    plt.ylabel("Volume (R$)")
    plt.xlabel("Strike")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()


def plot_gex_friday(fri_gex_by_strike, spot, underlying, next_friday_str,
                    fri_dte, fri_flip, fri_call_wall, fri_put_wall,
                    show_plots=False):
    """Bar chart of GEX for options expiring next Friday."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_axisbelow(True)
    fri_s = fri_gex_by_strike['Strike'].to_numpy(dtype=float)
    fri_g = (fri_gex_by_strike['GEX_customer'] / 1e6).to_numpy(dtype=float)

    u_fri = np.unique(fri_s)
    if len(u_fri) >= 3:
        bw = np.median(np.diff(u_fri)) * 0.6
    elif len(u_fri) == 2:
        bw = abs(u_fri[1] - u_fri[0]) * 0.6
    else:
        bw = 0.1

    colors = np.where(fri_g >= 0, "#10B981", "#EF4444")
    ax.bar(fri_s, fri_g, width=bw, color=colors,
           edgecolor="none", alpha=0.6, zorder=3)

    if len(fri_g) > 2:
        sm = pd.Series(fri_g).rolling(3, center=True, min_periods=1).mean().values
        ax.plot(fri_s, sm, color='#3B82F6', lw=2, zorder=4, label='Smoothed GEX')

    ax.axvline(spot, color='green', lw=1.2, zorder=5, label=f'Spot: {spot:.2f}')
    if np.isfinite(fri_flip):
        ax.axvline(fri_flip, color='#F59E0B', lw=1.2, ls='--', zorder=5,
                   label=f"Flip: {fri_flip:.2f}")
    if np.isfinite(fri_call_wall):
        ax.axvline(fri_call_wall, color='#2563EB', ls=':', lw=1.6,
                   label=f"Call Wall: {fri_call_wall:.2f}")
        ax.annotate(f"Call Wall\n{fri_call_wall:.2f}",
                    xy=(fri_call_wall, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1),
                    xytext=(8, -18), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#2563EB',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2563EB', alpha=0.85),
                    ha='left', va='top')
    if np.isfinite(fri_put_wall):
        ax.axvline(fri_put_wall, color='#DC2626', ls='--', lw=1.6,
                   label=f"Put Wall: {fri_put_wall:.2f}")
        ax.annotate(f"Put Wall\n{fri_put_wall:.2f}",
                    xy=(fri_put_wall, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1),
                    xytext=(-8, 18), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#DC2626',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#DC2626', alpha=0.85),
                    ha='right', va='bottom')

    cw_str = f"{fri_call_wall:.2f}" if np.isfinite(fri_call_wall) else "N/A"
    pw_str = f"{fri_put_wall:.2f}" if np.isfinite(fri_put_wall) else "N/A"
    ax.set_title(f"{underlying} — GEX Next Friday ({next_friday_str}, {fri_dte} DTE)"
                 f"  |  Call Wall: {cw_str}  |  Put Wall: {pw_str}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('GEX (millions)')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_gex_all_expiry(gex_by_strike, spot, underlying, gamma_flip,
                        call_wall, put_wall, show_plots=False, save_path=None):
    """Render a dark SPX-style GEX snapshot for B3 assets."""
    if gex_by_strike is None or gex_by_strike.empty:
        print(f"[!] No GEX data to plot for {underlying} - skipping snapshot chart.")
        return

    strikes = gex_by_strike['Strike'].to_numpy(dtype=float)
    gvals = (gex_by_strike['GEX_customer'] / 1e6).to_numpy(dtype=float)
    valid = np.isfinite(strikes) & np.isfinite(gvals)
    strikes = strikes[valid]
    gvals = gvals[valid]
    if len(strikes) == 0:
        print(f"[!] No valid GEX points to plot for {underlying} - skipping snapshot chart.")
        return

    u = np.unique(strikes)
    if len(u) >= 3:
        step = np.median(np.diff(u))
    elif len(u) == 2:
        step = abs(u[1] - u[0])
    else:
        step = max(spot * 0.0025, 0.1)
    bar_width = step * 0.65

    smooth = pd.Series(gvals).rolling(5, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(13, 7), facecolor="#05070B")
    ax.set_facecolor("#05070B")
    ax.set_axisbelow(True)

    bar_colors = np.where(gvals >= 0, "#2DD4BF", "#EF4444")
    bars = ax.bar(
        strikes, gvals, width=bar_width, align="center",
        color=bar_colors, edgecolor="none", alpha=0.78, zorder=3,
        label="GEX $M"
    )

    smooth_line, = ax.plot(
        strikes, smooth, color="#2563EB", lw=2.2, zorder=4,
        label="Agg Gamma (Smooth)"
    )

    spot_line = ax.axvline(
        spot, color="#F9FAFB", lw=1.6, ls=(0, (4, 4)), zorder=5,
        label="Spot Price"
    )

    flip_line = None
    if np.isfinite(gamma_flip):
        flip_line = ax.axvline(
            gamma_flip, color="#F59E0B", lw=1.6, ls=(0, (4, 4)), zorder=5,
            label="Gamma Flip"
        )

    call_line = None
    if np.isfinite(call_wall):
        call_line = ax.axvline(
            call_wall, color="#2563EB", lw=1.6, ls=(0, (4, 4)), zorder=5,
            label="Call Wall"
        )

    put_line = None
    if np.isfinite(put_wall):
        put_line = ax.axvline(
            put_wall, color="#DC2626", lw=1.6, ls=(0, (4, 4)), zorder=5,
            label="Put Wall"
        )

    ymin = float(np.nanmin(gvals)) if len(gvals) else -1.0
    ymax = float(np.nanmax(gvals)) if len(gvals) else 1.0
    if ymin < 0 < ymax:
        lim = max(abs(ymin), abs(ymax)) * 1.18
        ax.set_ylim(-lim, lim)
    else:
        pad = 0.15 * (ymax - ymin if ymax > ymin else max(1.0, abs(ymax)))
        ax.set_ylim(ymin - pad, ymax + pad)

    anchors = [v for v in [spot, gamma_flip, call_wall, put_wall] if np.isfinite(v)]
    if anchors:
        left = min(anchors) * 0.97
        right = max(anchors) * 1.03
        if right <= left:
            right = left + max(step * 10, 1.0)
        ax.set_xlim(left, right)

    ax.axhline(0, color="#6B7280", lw=0.9, alpha=0.8, zorder=2)
    ax.grid(axis="y", color="#1F2937", alpha=0.8, linewidth=0.7)

    for spine in ax.spines.values():
        spine.set_color("#111827")
    ax.tick_params(colors="#E5E7EB", labelsize=10)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Strike Price", color="#F3F4F6", fontsize=12, fontweight="bold")
    ax.set_ylabel("GEX ($ Millions)", color="#F3F4F6", fontsize=12, fontweight="bold")

    snapshot_date = pd.Timestamp.now().strftime("%d %b %Y")
    flip_str = f"{gamma_flip:,.2f}" if np.isfinite(gamma_flip) else "N/A"
    cw_str = f"{call_wall:,.2f}" if np.isfinite(call_wall) else "N/A"
    pw_str = f"{put_wall:,.2f}" if np.isfinite(put_wall) else "N/A"
    ax.set_title(
        f"{underlying} GEX snapshot: {snapshot_date} • current+next week\n"
        f"Flip: {flip_str} | Walls: {cw_str}/{pw_str} | local B3 analytics",
        loc="left", color="#F9FAFB", fontsize=15, fontweight="bold", pad=16
    )
    fig.text(0.92, 0.952, "local B3 proxy", ha="right", va="center",
             fontsize=9, color="#9CA3AF")

    handles = [bars, smooth_line, spot_line]
    if flip_line is not None:
        handles.append(flip_line)
    if call_line is not None:
        handles.append(call_line)
    if put_line is not None:
        handles.append(put_line)

    legend = ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(6, len(handles)), fontsize=9, framealpha=0.95,
        facecolor="#0B1220", edgecolor="#1F2937"
    )
    for text in legend.get_texts():
        text.set_color("#F3F4F6")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[OK] Saved GEX snapshot chart: {save_path}")

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_gex_weekly(weekly_results, spot, underlying, show_plots=False):
    """
    Side-by-side bar charts of GEX for current-week and next-week expirations.

    Parameters
    ----------
    weekly_results : list[dict]
        Output of ``compute_weekly_walls`` — one dict per week.
    spot : float
        Current underlying spot price.
    underlying : str
        Ticker label for chart titles.
    show_plots : bool
        If False, suppress plt.show().
    """
    n_panels = len([w for w in weekly_results if not w['gex_by_strike'].empty])
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(14 * n_panels / 2, 6),
                             squeeze=False)
    axes = axes.ravel()
    panel = 0

    for wk in weekly_results:
        gex_df = wk['gex_by_strike']
        if gex_df.empty:
            continue

        ax = axes[panel]
        ax.set_axisbelow(True)

        strikes = gex_df['Strike'].to_numpy(dtype=float)
        gvals = (gex_df['GEX_customer'] / 1e6).to_numpy(dtype=float)

        u = np.unique(strikes)
        if len(u) >= 3:
            bw = np.median(np.diff(u)) * 0.6
        elif len(u) == 2:
            bw = abs(u[1] - u[0]) * 0.6
        else:
            bw = 0.1

        colors = np.where(gvals >= 0, "#10B981", "#EF4444")
        ax.bar(strikes, gvals, width=bw, color=colors,
               edgecolor="none", alpha=0.6, zorder=3)

        if len(gvals) > 2:
            sm = pd.Series(gvals).rolling(3, center=True, min_periods=1).mean().values
            ax.plot(strikes, sm, color='#3B82F6', lw=2, zorder=4,
                    label='Smoothed GEX')

        ax.axvline(spot, color='green', lw=1.2, zorder=5,
                   label=f'Spot: {spot:.2f}')

        flip = wk['gamma_flip']
        cw = wk['call_wall']
        pw = wk['put_wall']

        if np.isfinite(flip):
            ax.axvline(flip, color='#F59E0B', lw=1.2, ls='--', zorder=5,
                       label=f"Flip: {flip:.2f}")
        if np.isfinite(cw):
            ax.axvline(cw, color='#2563EB', ls=':', lw=1.6,
                       label=f"Call Wall: {cw:.2f}")
            ax.annotate(f"Call Wall\n{cw:.2f}",
                        xy=(cw, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1),
                        xytext=(8, -18), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#2563EB',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='#2563EB', alpha=0.85),
                        ha='left', va='top')
        if np.isfinite(pw):
            ax.axvline(pw, color='#DC2626', ls='--', lw=1.6,
                       label=f"Put Wall: {pw:.2f}")
            ax.annotate(f"Put Wall\n{pw:.2f}",
                        xy=(pw, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1),
                        xytext=(-8, 18), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#DC2626',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='#DC2626', alpha=0.85),
                        ha='right', va='bottom')

        cw_s = f"{cw:.2f}" if np.isfinite(cw) else "N/A"
        pw_s = f"{pw:.2f}" if np.isfinite(pw) else "N/A"
        ax.set_title(
            f"{underlying} — {wk['label']} ({wk['friday_str']}, {wk['dte']} BD)\n"
            f"Call Wall: {cw_s}  |  Put Wall: {pw_s}",
            fontsize=11, fontweight='bold')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('GEX (millions)')
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25)
        panel += 1

    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close(fig)

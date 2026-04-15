PERIODS = 3
SHIFT_PERIODS = 0
MAGIC_NUMBER = 345346
GEX_MAGIC_NUMBER = 345347          # Separate magic for GEX pending orders
CALL_OPTION = 0
UNIX_DAYS_IN_SECONDS = 60*60*24
MIN_DAYS_TO_EXPIRY = 35*UNIX_DAYS_IN_SECONDS # 45 days in seconds
ASSET_SYMBOL = ["BOVA11", "VALE3","HASH11","PETR4","ITUB4","BBAS3"] #, "GOAU4", "BBAS3", "BRAV3", "ITUB4", "BBDC4", "MGLU3", "RAIZ4"]
PLOT_GEX = False  # Set to False to skip all GEX chart generation

# --- GEX Order & Monitor Settings ---
GEX_SEND_ORDERS = True            # Set to True to enable order execution
GEX_ORDER_VOLUME = 1.0             # Min volume per order (fallback if margin calc fails)
GEX_ORDER_DEVIATION = 5            # Price deviation allowed (points)
GEX_MARGIN_FREE_PCT = 0.05        # % of free margin as total GEX budget (0.01 = 1%)
GEX_SL_RISK_PCT = 0.40            # Stop loss as % of total margin (0.40 = 40% → R$200 on R$500)
GEX_TRAILING_ACTIVATION_PCT = 0.30 # Trailing stop activates at 30% of margin profit (R$150)
GEX_DCA_LOSS_STEP_PCT = 0.10      # Add new order every 10% margin loss (R$50 steps)
GEX_DCA_MAX_ORDERS = 3            # Max DCA additions per side (total = initial + 3)
GEX_MIN_SIGNAL_STRENGTH = 2       # Minimum signal strength to place order (0-3)
GEX_WALL_PROXIMITY_PCT = -0.001    # Entry offset from S/R zone (slightly widened for mapper noise)
GEX_MONITOR_INTERVAL = 5          # Spot price poll interval in seconds (faster for tighter trailing)
GEX_MONITOR_ENABLED = False       # Set to True to start real-time GEX monitor after analysis
GEX_RTD_REFRESH_INTERVAL = 300   # Re-read RTD OI file every N seconds (0 = disabled)

# --- Risk & Profit Management ---
GEX_MIN_SL_POINTS = 300           # Minimum SL distance (points) — floor after DCA compression
GEX_TRAILING_DISTANCE_FACTOR = 0.75 # Trail at 75% of original SL distance once activated
GEX_MAX_DAILY_LOSS_PCT = 0.50     # Halt new entries if daily loss exceeds 50% of budget
GEX_TP_AT_OPPOSITE_WALL = True    # Set TP at opposite GEX wall (call wall for BUY, put wall for SELL)
GEX_TRADE_WINDOW_START = "11:00"  # Earliest entry time (BRT) — skip auction + first-hour noise
GEX_TRADE_WINDOW_END = "16:30"    # Latest entry time (BRT) — skip low-liquidity close
GEX_PRE_TRADE_REFRESH_MIN = 15    # Minutes before trading window to recalculate GEX levels
PERIODS = 3
SHIFT_PERIODS = 0
MAGIC_NUMBER = 345346
GEX_MAGIC_NUMBER = 345347          # Separate magic for GEX pending orders
CALL_OPTION = 0
UNIX_DAYS_IN_SECONDS = 60*60*24
MIN_DAYS_TO_EXPIRY = 35*UNIX_DAYS_IN_SECONDS # 45 days in seconds
ASSET_SYMBOL = ["BOVA11", "VALE3", "PETR4"] #, "GOAU4", "BBAS3", "BRAV3", "ITUB4", "BBDC4", "MGLU3", "RAIZ4"]
PLOT_GEX = False  # Set to False to skip all GEX chart generation

# --- GEX Order & Monitor Settings ---
GEX_SEND_ORDERS = True            # Set to True to enable order execution
GEX_ORDER_VOLUME = 1.0             # Min volume per order (fallback if margin calc fails)
GEX_ORDER_DEVIATION = 5            # Price deviation allowed (points)
GEX_MARGIN_FREE_PCT = 0.05        # % of free margin as total GEX budget (0.01 = 1%)
GEX_SL_RISK_PCT = 0.50            # Stop loss as % of total margin (0.50 = 50%)
GEX_TRAILING_ACTIVATION_PCT = 0.35 # Trailing stop activates at 35% of margin profit (R$175)
GEX_DCA_LOSS_STEP_PCT = 0.10      # Add new order every 10% margin loss (R$50 steps)
GEX_DCA_MAX_ORDERS = 4            # Max DCA additions per side (total = initial + 4)
GEX_MIN_SIGNAL_STRENGTH = 2       # Minimum signal strength to place order (0-3)
GEX_WALL_PROXIMITY_PCT = 0.001    # % distance from wall for entry zone (0.01 = 1%)
GEX_MONITOR_INTERVAL = 10         # Spot price poll interval in seconds
GEX_MONITOR_ENABLED = True       # Set to True to start real-time GEX monitor after analysis
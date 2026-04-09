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
GEX_ORDER_VOLUME = 1.0             # Volume per order (mini contracts for WIN)
GEX_ORDER_DEVIATION = 5            # Price deviation allowed (points)
GEX_MIN_SIGNAL_STRENGTH = 2       # Minimum signal strength to place order (0-3)
GEX_WALL_PROXIMITY_PCT = 0.01    # % distance from wall for entry zone (0.015 = 1.5%)
GEX_MONITOR_INTERVAL = 10         # Spot price poll interval in seconds
GEX_MONITOR_ENABLED = True       # Set to True to start real-time GEX monitor after analysis
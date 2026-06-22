PERIODS = 3
SHIFT_PERIODS = 0
MAGIC_NUMBER = 345346
GEX_MAGIC_NUMBER = 345347          # Separate magic for GEX pending orders
CALL_OPTION = 0
UNIX_DAYS_IN_SECONDS = 60*60*24
MIN_DAYS_TO_EXPIRY = 35*UNIX_DAYS_IN_SECONDS # 45 days in seconds
ASSET_SYMBOL = ["BOVA11","BRKM5","VALE3","HASH11","PETR4","ITUB4","BBAS3"] #, "GOAU4", "BBAS3", "BRAV3", "ITUB4", "BBDC4", "MGLU3", "RAIZ4"]
PLOT_GEX = False  # Set to False to skip all GEX chart generation

# --- TRADING MODE SELECTOR ---
# Choose "CONSERVATIVE" for low-risk defaults, "MODERATE" for balanced risk
TRADING_MODE = "CONSERVATIVE"

# --- GEX Order & Monitor Settings (Base) ---
GEX_SEND_ORDERS = True            # Set to True to enable order execution
GEX_ORDER_DEVIATION = 5            # Price deviation allowed (points)
GEX_MONITOR_INTERVAL = 5          # Spot price poll interval in seconds (faster for tighter trailing)
GEX_MONITOR_ENABLED = False       # Set to True to start real-time GEX monitor after analysis

# --- Confirmation & Setup Filters (Fixed across all modes) ---
GEX_REQUIRE_5M_CONFIRMATION = True  # Require 5-minute directional confirmation before entry
GEX_CONFIRMATION_MINUTES = 5        # Confirmation window in minutes
GEX_NEUTRAL_ONLY = False            # Only enter in neutral setup (between walls and near gamma flip)
GEX_NEUTRAL_MAX_FLIP_DISTANCE_PCT = 0.005  # Max distance to flip for neutral setup (0.5%)

# --- PROFILE: CONSERVATIVE (Low risk, smaller size, fewer DCA) ---
_PROFILE_CONSERVATIVE = {
    'order_volume': 1.0,            # Minimum for WIN futures (integer lots only)
    'margin_free_pct': 0.02,        # Use only 2% of free margin as total budget
    'sl_risk_pct': 0.30,            # Stop loss as 30% of margin (tighter)
    'dca_max_orders': 1,            # Only 1 DCA addition (total = initial + 1)
    'min_signal_strength': 2,       # Require full signal strength
    'wall_proximity_pct': 0.005,    # Bracket the wall (±0.5%) so spot can land in zone
    'trailing_activation_pct': 0.30,# Trail at 30% profit
}

# --- PROFILE: MODERATE (Balanced risk, standard sizing) ---
_PROFILE_MODERATE = {
    'order_volume': 1.0,            # Standard initial size
    'margin_free_pct': 0.05,        # Use 15% of free margin as total budget
    'sl_risk_pct': 0.40,            # Stop loss as 40% of margin
    'dca_max_orders': 3,            # Up to 3 DCA additions (total = initial + 3)
    'min_signal_strength': 2,       # Require full signal strength
    'wall_proximity_pct': 0.005,    # Bracket the wall (±0.5%) so spot can land in zone
    'trailing_activation_pct': 0.30,# Trail at 30% profit
}

# --- Apply selected profile ---
_profile = _PROFILE_CONSERVATIVE if TRADING_MODE == "CONSERVATIVE" else _PROFILE_MODERATE

GEX_ORDER_VOLUME = _profile['order_volume']
GEX_MARGIN_FREE_PCT = _profile['margin_free_pct']
GEX_SL_RISK_PCT = _profile['sl_risk_pct']
GEX_DCA_MAX_ORDERS = _profile['dca_max_orders']
GEX_MIN_SIGNAL_STRENGTH = _profile['min_signal_strength']
GEX_WALL_PROXIMITY_PCT = _profile['wall_proximity_pct']
GEX_TRAILING_ACTIVATION_PCT = _profile['trailing_activation_pct']

# --- Risk & Profit Management (Shared) ---
GEX_DCA_LOSS_STEP_PCT = 0.10      # Add new order every 10% margin loss (R$50 steps)
GEX_MIN_SL_POINTS = 300           # Minimum SL distance (points) — floor after DCA compression
GEX_TRAILING_DISTANCE_FACTOR = 0.75 # Trail at 75% of original SL distance once activated
GEX_MAX_DAILY_LOSS_PCT = 0.50     # Halt new entries if daily loss exceeds 50% of budget
GEX_TP_AT_OPPOSITE_WALL = True    # Set TP at opposite GEX wall (call wall for BUY, put wall for SELL)
GEX_TRADE_WINDOW_START = "11:00"  # Earliest entry time (BRT) — skip auction + first-hour noise
GEX_TRADE_WINDOW_END = "16:30"    # Latest entry time (BRT) — skip low-liquidity close
GEX_PRE_TRADE_REFRESH_MIN = 15    # Minutes before trading window to recalculate GEX levels
FUTURES_ROLL_BUFFER_HOURS = 24    # Skip contracts expiring within this many hours (roll to next)
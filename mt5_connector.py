import bisect
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import time
import pandas as pd

from constants import CALL_OPTION, MAGIC_NUMBER, MIN_DAYS_TO_EXPIRY, PERIODS, SHIFT_PERIODS
from constants import GEX_MAGIC_NUMBER


def _normalize_to_tick(price: float, tick_size: float, digits: int) -> float:
    if tick_size and tick_size > 0:
        return round(round(price / tick_size) * tick_size, digits)
    return round(price, digits)


def _sanitize_stops(symbol, symbol_info, order_type, sl, tp):
    """Adjust SL/TP so the broker accepts them.

    TRADE_RETCODE_INVALID_STOPS (10016) fires when SL/TP are:
      - on the wrong side of the current bid/ask for the order direction, or
      - closer than ``trade_stops_level`` (or ``trade_freeze_level``) points
        from the current price, or
      - not aligned to ``trade_tick_size``.

    Strategy: drop a stop that's on the wrong side (set to 0); push a
    same-side but too-close stop to the minimum allowed distance. Always
    normalize to tick size.
    """
    sl = float(sl or 0.0)
    tp = float(tp or 0.0)
    if sl == 0.0 and tp == 0.0:
        return 0.0, 0.0

    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.bid <= 0 or tick.ask <= 0:
        return sl, tp  # can't validate without a quote; let server decide

    point = symbol_info.point or 0.0
    tick_size = symbol_info.trade_tick_size or point or 0.0
    digits = symbol_info.digits

    stops_level = max(
        getattr(symbol_info, "trade_stops_level", 0) or 0,
        getattr(symbol_info, "trade_freeze_level", 0) or 0,
    )
    min_dist = stops_level * point  # distance in price units

    is_buy = order_type == mt5.ORDER_TYPE_BUY
    bid = tick.bid
    ask = tick.ask

    # --- Stop loss --------------------------------------------------------
    if sl > 0.0:
        if is_buy:
            max_sl = bid - min_dist
            if sl >= bid:           # wrong side
                print(f"[STOPS] {symbol}: SL {sl:.{digits}f} >= bid {bid:.{digits}f} (wrong side) — dropped")
                sl = 0.0
            elif sl > max_sl:        # too close
                new_sl = _normalize_to_tick(max_sl, tick_size, digits)
                print(f"[STOPS] {symbol}: SL {sl:.{digits}f} too close (min dist {min_dist}); pushed to {new_sl:.{digits}f}")
                sl = new_sl
            else:
                sl = _normalize_to_tick(sl, tick_size, digits)
        else:  # SELL
            min_sl = ask + min_dist
            if sl <= ask:
                print(f"[STOPS] {symbol}: SL {sl:.{digits}f} <= ask {ask:.{digits}f} (wrong side) — dropped")
                sl = 0.0
            elif sl < min_sl:
                new_sl = _normalize_to_tick(min_sl, tick_size, digits)
                print(f"[STOPS] {symbol}: SL {sl:.{digits}f} too close (min dist {min_dist}); pushed to {new_sl:.{digits}f}")
                sl = new_sl
            else:
                sl = _normalize_to_tick(sl, tick_size, digits)

    # --- Take profit ------------------------------------------------------
    if tp > 0.0:
        if is_buy:
            min_tp = ask + min_dist
            if tp <= ask:
                print(f"[STOPS] {symbol}: TP {tp:.{digits}f} <= ask {ask:.{digits}f} (wrong side) — dropped")
                tp = 0.0
            elif tp < min_tp:
                new_tp = _normalize_to_tick(min_tp, tick_size, digits)
                print(f"[STOPS] {symbol}: TP {tp:.{digits}f} too close (min dist {min_dist}); pushed to {new_tp:.{digits}f}")
                tp = new_tp
            else:
                tp = _normalize_to_tick(tp, tick_size, digits)
        else:  # SELL
            max_tp = bid - min_dist
            if tp >= bid:
                print(f"[STOPS] {symbol}: TP {tp:.{digits}f} >= bid {bid:.{digits}f} (wrong side) — dropped")
                tp = 0.0
            elif tp > max_tp:
                new_tp = _normalize_to_tick(max_tp, tick_size, digits)
                print(f"[STOPS] {symbol}: TP {tp:.{digits}f} too close (min dist {min_dist}); pushed to {new_tp:.{digits}f}")
                tp = new_tp
            else:
                tp = _normalize_to_tick(tp, tick_size, digits)

    return sl, tp


def _sanitize_pending_stops(symbol_info, market_type, ref_price, sl, tp):
    """Same idea as _sanitize_stops, but for pending orders the reference
    price is the order's own limit price (not bid/ask)."""
    sl = float(sl or 0.0)
    tp = float(tp or 0.0)
    if sl == 0.0 and tp == 0.0:
        return 0.0, 0.0

    point = symbol_info.point or 0.0
    tick_size = symbol_info.trade_tick_size or point or 0.0
    digits = symbol_info.digits
    stops_level = max(
        getattr(symbol_info, "trade_stops_level", 0) or 0,
        getattr(symbol_info, "trade_freeze_level", 0) or 0,
    )
    min_dist = stops_level * point
    is_buy = market_type == mt5.ORDER_TYPE_BUY

    if sl > 0.0:
        if is_buy:
            limit = ref_price - min_dist
            if sl >= ref_price:
                sl = 0.0
            elif sl > limit:
                sl = _normalize_to_tick(limit, tick_size, digits)
            else:
                sl = _normalize_to_tick(sl, tick_size, digits)
        else:
            limit = ref_price + min_dist
            if sl <= ref_price:
                sl = 0.0
            elif sl < limit:
                sl = _normalize_to_tick(limit, tick_size, digits)
            else:
                sl = _normalize_to_tick(sl, tick_size, digits)

    if tp > 0.0:
        if is_buy:
            limit = ref_price + min_dist
            if tp <= ref_price:
                tp = 0.0
            elif tp < limit:
                tp = _normalize_to_tick(limit, tick_size, digits)
            else:
                tp = _normalize_to_tick(tp, tick_size, digits)
        else:
            limit = ref_price - min_dist
            if tp >= ref_price:
                tp = 0.0
            elif tp > limit:
                tp = _normalize_to_tick(limit, tick_size, digits)
            else:
                tp = _normalize_to_tick(tp, tick_size, digits)

    return sl, tp


class MT5Connector:

    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1
    TIMEFRAME_M15 = mt5.TIMEFRAME_M15
    TIMEFRAME_MN1 = getattr(mt5, 'TIMEFRAME_MN1', mt5.TIMEFRAME_D1)
    
    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MetaTrader 5")
        self.logger = logging.getLogger(__name__)

    def get_data(self, symbol, timeframe, periods, shift):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, shift, periods)
        if rates is None:
            self.logger.error(f"Could not get rates for {symbol}")
            return None
        else:
            df = pd.DataFrame(rates)
            df = df.dropna()
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Filter out weekends (keep only weekdays)
            df = df[df['time'].dt.weekday < 5]  # 0=Monday, ..., 4=Friday
            return df   

    def place_order(self, symbol, order_type, volume, price, deviation, comment,
                    magic=None, sl=0.0, tp=0.0):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Symbol {symbol} not found")
            return

        # --- Sanitize SL/TP to avoid TRADE_RETCODE_INVALID_STOPS (10016) ---
        sl, tp = _sanitize_stops(symbol, symbol_info, order_type, sl, tp)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic if magic is not None else MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            print(f"Order send failed, error: {mt5.last_error()}")
        else:
            print(f"Order send result: retcode={result.retcode}, comment={result.comment}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Order executed successfully! Order: {result.order}, Deal: {result.deal}")
            elif result.retcode == mt5.TRADE_RETCODE_INVALID_STOPS and (sl or tp):
                # Last-resort retry without SL/TP so the position at least opens;
                # the trailing/monitor loop will set a stop on the next tick.
                print(f"Order failed: {result.comment} — retrying without SL/TP")
                request["sl"] = 0.0
                request["tp"] = 0.0
                result = mt5.order_send(request)
                if result is None:
                    print(f"Retry failed, error: {mt5.last_error()}")
                elif result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Order executed (no SL/TP) Order: {result.order}, Deal: {result.deal}")
                else:
                    print(f"Retry failed: {result.comment} (retcode={result.retcode})")
            else:
                print(f"Order failed: {result.comment}")

        return result

    def cancel_gex_pending_orders(self, symbol=None):
        """Cancel all pending orders placed by GEX (identified by GEX_MAGIC_NUMBER).
        If symbol is given, only cancel orders for that symbol."""
        orders = mt5.orders_get()
        if orders is None or len(orders) == 0:
            return 0

        cancelled = 0
        for order in orders:
            if order.magic != GEX_MAGIC_NUMBER:
                continue
            if symbol is not None and order.symbol != symbol:
                continue

            cancel_request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
            }
            result = mt5.order_send(cancel_request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[GEX] Cancelled pending order #{order.ticket} "
                      f"({order.symbol} {'BUY_LIMIT' if order.type == mt5.ORDER_TYPE_BUY_LIMIT else 'SELL_LIMIT'} "
                      f"@ {order.price_open:.0f})")
                cancelled += 1
            else:
                err = result.comment if result else mt5.last_error()
                print(f"[GEX] Failed to cancel order #{order.ticket}: {err}")

        return cancelled

    def place_pending_order(self, symbol, order_type, volume, price, deviation, comment,
                            sl=0.0, tp=0.0):
        """Place a pending limit order (BUY_LIMIT or SELL_LIMIT).

        Parameters
        ----------
        symbol : str        Target symbol (e.g. 'WINM26')
        order_type : int    mt5.ORDER_TYPE_BUY_LIMIT or mt5.ORDER_TYPE_SELL_LIMIT
        volume : float      Number of contracts
        price : float       Limit price
        deviation : int     Max allowed deviation in points
        comment : str       Order comment
        sl, tp : float      Stop-loss / take-profit (0 = none)

        Returns
        -------
        result or None
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[GEX] Symbol {symbol} not found")
            return None

        if not mt5.symbol_select(symbol, True):
            print(f"[GEX] Failed to select symbol {symbol}")
            return None

        # Normalize price to symbol tick size
        tick_size = symbol_info.trade_tick_size
        if tick_size > 0:
            price = round(round(price / tick_size) * tick_size, symbol_info.digits)

        # Sanitize SL/TP using the limit price as the reference. Pending-order
        # validation uses the same stops_level distance from the order price.
        market_type = mt5.ORDER_TYPE_BUY if order_type == mt5.ORDER_TYPE_BUY_LIMIT else mt5.ORDER_TYPE_SELL
        sl, tp = _sanitize_pending_stops(symbol_info, market_type, price, sl, tp)

        # Determine filling mode
        filling = symbol_info.filling_mode
        if filling & 2:
            type_filling = mt5.ORDER_FILLING_FOK
        elif filling & 1:
            type_filling = mt5.ORDER_FILLING_IOC
        else:
            type_filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": GEX_MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_DAY,      # Expires end of day
            "type_filling": type_filling,
        }

        # Pre-check
        check = mt5.order_check(request)
        if check is None:
            print(f"[GEX] order_check failed for {symbol}, error: {mt5.last_error()}")
            return None
        if check.retcode != 0:
            print(f"[GEX] order_check rejected: retcode={check.retcode}, {check.comment}")
            return None

        result = mt5.order_send(request)
        if result is None:
            print(f"[GEX] order_send failed, error: {mt5.last_error()}")
        else:
            type_str = "BUY_LIMIT" if order_type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL_LIMIT"
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[GEX] {type_str} placed: {symbol} {volume} @ {price:.0f}  "
                      f"order #{result.order}")
            else:
                print(f"[GEX] {type_str} failed: {result.comment} (retcode={result.retcode})")
        return result

    def place_order_vertical(self,symbolY,symbolX,orders_type,volume,iv_y,iv_x):
        print(f"Placing vertical order...{symbolY}, {symbolX}, {orders_type}, {volume}, {iv_y}, {iv_x}")

        
        # Enable symbols in Market Watch
        if not mt5.symbol_select(symbolY, True):
            print(f"Failed to select symbol {symbolY}")
            return
        
        if not mt5.symbol_select(symbolX, True):
            print(f"Failed to select symbol {symbolX}")
            return
        
        # Get symbol info
        symbol_info_y = mt5.symbol_info(symbolY)
        symbol_info_x = mt5.symbol_info(symbolX)
        
        if symbol_info_y is None:
            print(f"Symbol {symbolY} not found")
            return
        
        if symbol_info_x is None:
            print(f"Symbol {symbolX} not found")
            return

        print(f"Symbol info X: Ask={symbol_info_x.ask}, Bid={symbol_info_x.bid}, Min={symbol_info_x.volume_min}, Max={symbol_info_x.volume_max}")

        
        # Get the filling mode supported by the symbol
        filling_type_y = symbol_info_y.filling_mode
        filling_type_x = symbol_info_x.filling_mode
        
        # Determine appropriate filling mode for symbol Y
        if filling_type_y & 2:  # FOK is supported
            type_filling_y = mt5.ORDER_FILLING_FOK
        elif filling_type_y & 1:  # IOC is supported
            type_filling_y = mt5.ORDER_FILLING_IOC
        else:  # Return is supported
            type_filling_y = mt5.ORDER_FILLING_RETURN
        
        # Determine appropriate filling mode for symbol X
        if filling_type_x & 1:  # IOC is supported
            type_filling_x = mt5.ORDER_FILLING_IOC
        elif filling_type_x & 4:  # Return is supported
            type_filling_x = mt5.ORDER_FILLING_RETURN
        else:  # FOK is supported
            type_filling_x = mt5.ORDER_FILLING_FOK
        
        print(f"Placing orders for symbols {symbolY} and {symbolX} and order type {orders_type[0]} and order type {orders_type[1]} respectively")

        # Prepare the first request (Y symbol)
        request_y = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolY,
           "volume": volume,
           "type": orders_type[0],
           "price": symbol_info_y.ask,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": MAGIC_NUMBER,
           "comment": "y,{:.2f}".format(iv_y),
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": type_filling_y,
        }
        
        result_y_check = mt5.order_check(request_y)

        if result_y_check is None:
            print(f"order_check failed for {symbolY}, error: {mt5.last_error()}")
            return
        else:
            print(f"Order check Y result: retcode={result_y_check.retcode}, comment={result_y_check.comment}")
            # retcode 0 means check passed successfully
            if result_y_check.retcode != 0:
                print(f"Order check Y failed: retcode={result_y_check.retcode}, {result_y_check.comment}")
                return

        # Prepare the second request (X symbol)
        request_x = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolX,
           "volume": volume,
           "type": orders_type[1],
           "price": symbol_info_x.bid,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": MAGIC_NUMBER,
           "comment": "x,{:.2f}".format(iv_x),
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": type_filling_x,
        }
        
        result_x_order_check = mt5.order_check(request_x)

        if result_x_order_check is None:
            print(f"order_check failed for {symbolX}, error: {mt5.last_error()}")
            return
        else:
            print(f"Order check X result: retcode={result_x_order_check.retcode}, comment={result_x_order_check.comment}")
            # retcode 0 means check passed successfully
            if result_x_order_check.retcode != 0:
                print(f"Order check X failed: retcode={result_x_order_check.retcode}, {result_x_order_check.comment}")
                return

        # Both checks passed (retcode == 0), now send orders
        print("Both order checks passed, sending orders...")
        result_y_order = mt5.order_send(request_y)
        result_x_order = mt5.order_send(request_x)

        if result_y_order is None:
            print(f"Order send Y failed, error: {mt5.last_error()}")
        else:
            print(f"Order send Y result: retcode={result_y_order.retcode}, comment={result_y_order.comment}")
            if result_y_order.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Order Y executed successfully! Order: {result_y_order.order}, Deal: {result_y_order.deal}")
            else:
                print(f"Order Y failed: {result_y_order.comment}")

        if result_x_order is None:
            print(f"Order send X failed, error: {mt5.last_error()}")
        else:
            print(f"Order send X result: retcode={result_x_order.retcode}, comment={result_x_order.comment}")
            if result_x_order.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Order X executed successfully! Order: {result_x_order.order}, Deal: {result_x_order.deal}")
            else:
                print(f"Order X failed: {result_x_order.comment}")
    
    def close_all_positions(self):
        # Get all open positions
        positions = mt5.positions_get()
        if positions is not None or len(positions) > 0:
            # Loop through each position and close it
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                volume = position.volume
                position_magic = position.magic
                position_type = position.type  # 0 for buy, 1 for sell

            # Determine the opposite order type to close the position
                if position_type == mt5.ORDER_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    zscore = mt5.symbol_info_tick(symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    zscore = mt5.symbol_info_tick(symbol).ask

            # Create a close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type,
                    "position": ticket,
                    "zscore": zscore,
                    "deviation": 20,
                    "magic": MAGIC_NUMBER,
                    "comment": "Close position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

            # Send the close request
                if (position_magic != MAGIC_NUMBER):
                    continue
                result = mt5.order_send(request)

            # Check the result
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to close position {ticket} on {symbol}, Error code: {result.retcode}")
                else:
                    print(f"Successfully closed position {ticket} on {symbol}")

    def get_symbol_futures(self, group_name, include_expiring=False):
        """Return the current (trading) futures contract for *group_name*.

        Returns
        -------
        (expiration_time, symbol_name)              when include_expiring=False
        ((expiration_time, symbol_name), exp_name)   when include_expiring=True
            exp_name is the expiring contract still alive but inside the roll
            buffer (useful as historical-data fallback), or None.
        """
        import re
        from constants import FUTURES_ROLL_BUFFER_HOURS
        futures_symbols = mt5.symbols_get(group_name)
        if not futures_symbols:
            raise RuntimeError(f"No symbols found matching '{group_name}'")

        time_now = int(time.time())
        # Skip contracts expiring within the roll buffer (handles expiration-day roll)
        roll_cutoff = time_now + int(FUTURES_ROLL_BUFFER_HOURS * 3600)
        next_symbols_fut = {}
        rolling_symbols_fut = {}
        past_symbols_fut = {}

        # Extract base name from group pattern (e.g. "*WIN*" -> "WIN")
        base = group_name.replace("*", "")
        # Pattern for real contracts: BASE + month code + 2-digit year (e.g. WINM26)
        contract_re = re.compile(
            rf'^{re.escape(base)}[FGHJKMNQUVXZ]\d{{2}}$', re.IGNORECASE
        )

        for s in futures_symbols:
            # Skip synthetic/continuous symbols (WIN$N, WIN$, WINFUT, etc.)
            if not contract_re.match(s.name):
                continue
            if s.expiration_time > roll_cutoff:
               next_symbols_fut[s.expiration_time] = s.name
            elif s.expiration_time > time_now:
               # Contract still alive but inside roll buffer (expiring today)
               rolling_symbols_fut[s.expiration_time] = s.name
            else:
               past_symbols_fut[s.expiration_time] = s.name

        if not next_symbols_fut:
            raise RuntimeError(
                f"No unexpired futures contracts found matching '{group_name}'. "
                f"Candidates filtered: {[s.name for s in futures_symbols]}"
            )

        sorted_next_futures = dict(sorted(next_symbols_fut.items()))
        current_symbol = list(sorted_next_futures.items())[0]

        if include_expiring:
            # Return the most recent rolling/expiring contract (if any)
            expiring_name = None
            if rolling_symbols_fut:
                sorted_rolling = sorted(rolling_symbols_fut.items(), reverse=True)
                expiring_name = sorted_rolling[0][1]
            return current_symbol, expiring_name

        return current_symbol

    def get_win_symbols(self):
        """Return (current_symbol, previous_symbol) for WIN futures.

        current_symbol  : the front-month contract (e.g. WINM26)
        previous_symbol : the rolling/expiring contract when available,
                          otherwise the most-recent expired contract.
        """
        import re
        from constants import FUTURES_ROLL_BUFFER_HOURS

        futures_symbols = mt5.symbols_get("*WIN*")
        if not futures_symbols:
            return None, None

        time_now = int(time.time())
        roll_cutoff = time_now + int(FUTURES_ROLL_BUFFER_HOURS * 3600)
        next_symbols_fut = {}
        rolling_symbols_fut = {}
        past_symbols_fut = {}
        contract_re = re.compile(r'^WIN[FGHJKMNQUVXZ]\d{2}$', re.IGNORECASE)

        for s in futures_symbols:
            if not contract_re.match(s.name):
                continue
            if s.expiration_time > roll_cutoff:
                next_symbols_fut[s.expiration_time] = s.name
            elif s.expiration_time > time_now:
                rolling_symbols_fut[s.expiration_time] = s.name
            else:
                past_symbols_fut[s.expiration_time] = s.name

        current_symbol = None
        if next_symbols_fut:
            sorted_next = sorted(next_symbols_fut.items())
            current_symbol = sorted_next[0][1]

        previous_symbol = None
        if rolling_symbols_fut:
            sorted_rolling = sorted(rolling_symbols_fut.items(), reverse=True)
            previous_symbol = sorted_rolling[0][1]
        elif past_symbols_fut:
            sorted_past = sorted(past_symbols_fut.items())
            previous_symbol = sorted_past[-1][1]

        return current_symbol, previous_symbol

    def get_historical_futures_data(self, group_pattern, timeframe, periods, shift=0):
        """Fetch historical data combining current + previous WIN contract.

        Returns a single DataFrame with bars from the previous contract
        followed by the current contract (de-duplicated by time, preferring
        the current contract for overlapping bars).
        """
        current_sym, prev_sym = self.get_win_symbols()
        if current_sym is None:
            raise RuntimeError(f"No active WIN contract found for {group_pattern}")

        df_current = self.get_data(current_sym, timeframe, periods, shift)

        if prev_sym:
            df_prev = self.get_data(prev_sym, timeframe, periods, shift)
        else:
            df_prev = None

        if df_current is None or df_current.empty:
            if df_prev is not None and not df_prev.empty:
                print(f"[i] Using only previous contract {prev_sym} ({len(df_prev)} bars)")
                return df_prev, prev_sym
            raise RuntimeError(f"No data for {current_sym}" +
                               (f" or {prev_sym}" if prev_sym else ""))

        if df_prev is not None and not df_prev.empty:
            # Keep previous bars that are BEFORE the earliest current bar
            earliest_current = df_current['time'].min()
            older_bars = df_prev[df_prev['time'] < earliest_current]
            if not older_bars.empty:
                combined = pd.concat([older_bars, df_current], ignore_index=True)
                combined = combined.sort_values('time').reset_index(drop=True)
                print(f"[i] Combined {prev_sym} ({len(older_bars)} older bars) + "
                      f"{current_sym} ({len(df_current)} bars) = {len(combined)} total")
                return combined, current_sym
            else:
                print(f"[i] Previous contract {prev_sym} has no bars before {current_sym}")

        print(f"[i] Using {current_sym} only ({len(df_current)} bars)")
        return df_current, current_sym
    
    def get_options_chain(self,group_name,option_type):
        server_info = mt5.account_info().server
        print(f"Connected to MT5 server: {server_info}")
        options_symbols = mt5.symbols_get(group_name)
        print(f"Type for option symbols {type(options_symbols)} options symbols for group {group_name}")
        time_now = int(time.time())
        options_names = []
        expiration_time = 0
        expiration_limit = time_now + MIN_DAYS_TO_EXPIRY #10 days ahead
        # get the last expiration time before expiration_limit
        for s in options_symbols:
            if  s.expiration_time > expiration_limit: #call options only
                expiration_time = s.expiration_time
                break

        for s in options_symbols:
            if s.option_right in [option_type] and s.expiration_time == expiration_time: 
                options_names.append(s.name)

        sorted_options_names = sorted(options_names)
        return sorted_options_names
           
    def get_option_name_by_strike(self,group_name,strike_price,option_type,expiration_time):
        options_symbols = mt5.symbols_get(group_name)
        
        for s in options_symbols:
            
            if s.option_right in [option_type] and s.option_strike == strike_price and s.expiration_time == expiration_time:
               print(f"Found option symbol {s.name} for strike price {strike_price}")
               return s.name
        return None        
    
    def total_daily_risk(self):
        from_date = datetime.now() - timedelta(hours=12,minutes=0)
        #get the number of deals in history
        to_date=datetime.now()
        print(f"From date {from_date} to date {to_date}")
        deals=mt5.history_deals_get(from_date, to_date) 
        total_profit = 0
        total_volume = 0.0
        highest_score = 0.0
        traded_zscore = 0.0
        if deals==None:   
                print("No deals , error code={}".format(mt5.last_error()))   
        elif len(deals) > 0:        
            for deal in deals:
                if (len(deal.comment) > 1):
                    comment_deal = deal.comment.split(",")
                    
                    if (comment_deal[0] == 'y') or (comment_deal[0] == 'x'):
                        traded_zscore = abs(float(comment_deal[1]))
                    if (traded_zscore > highest_score):
                        highest_score = traded_zscore
                total_profit = total_profit + deal.commission + deal.profit
                total_volume = total_volume + deal.volume

        return highest_score,total_profit,total_volume
    
    def get_symbol_info(self,symbol):
        symbol_info = mt5.symbol_info(symbol)
        return symbol_info

    def get_account_info(self):
        account_info = mt5.account_info()
        return account_info    

    def get_open_positions(self):
        positions = mt5.positions_get()
        return positions
    
    def get_total_volume(self):
        total_volume = 0.0
        positions = mt5.positions_get()
        if positions is not None:
            for pos in positions:
                total_volume += pos.volume
        return total_volume
    
    def get_total_positions(self):
        total_positions = mt5.positions_total()
        return total_positions
    
    def last_error():
        last_error = mt5.last_error()
        return last_error

    def sleep(self, seconds):
        time.sleep(seconds)

    def initialize(self):
        return mt5.initialize()

    def shutdown(self):
        mt5.shutdown()

    def get_profit(self):
        profit = mt5.account_info().profit
        return profit
    
    def symbol_select(self,symbol,enable):
        return mt5.symbol_select(symbol,enable)
    
    def get_call_option_name_list(self,group_name):
        server_info = mt5.account_info().server
        print(f"Connected to MT5 server: {server_info}")  
        options_symbols = mt5.symbols_get(group_name)
        time_now = int(time.time())
        options_call_names = {}
        expiration_limit = time_now + MIN_DAYS_TO_EXPIRY #10 days ahead
        print(f"Current time {time_now} and expiration limit {expiration_limit}")
        # get the first expiration time after expiration_limit
        for s in options_symbols:
            
            if s.option_right in [CALL_OPTION] and s.expiration_time > expiration_limit: #call options only
               options_call_names[s.expiration_time] = s.name
               break
        
        sorted_options_call_names = dict(sorted(options_call_names.items()))
        print(f"Sorted call options names: {sorted_options_call_names}")
        return list(sorted_options_call_names.values())
    
    def get_option_names_by_expiration_time(self,symbol):
        
        time_now = int(time.time())
        min_expiration = time_now + MIN_DAYS_TO_EXPIRY #10 days ahead
        symbol_prefix = symbol[:4]
        group = f"*{symbol_prefix}*,!{symbol}*"
        options_symbols = mt5.symbols_get(group)
        expiration_times = set()
        chain_expiration = {}
        
       # get the first expiration time after min_expiration")
        for s in options_symbols:
            
            if s.expiration_time > min_expiration and s.option_strike > 0:
               expiration_times.add(s.expiration_time)

                       
        sorted_expiration_times = list(dict.fromkeys(expiration_times))
        sorted_expiration_times.sort()

        filtered_names = [symbol.name for symbol in options_symbols if symbol.expiration_time == sorted_expiration_times[0] and symbol.option_strike > 0]

        chain_expiration[sorted_expiration_times[0]] = filtered_names

        return chain_expiration
        
    
    def get_mt5_connector(self):
        return mt5

    def get_put_option_name_list(self,symbol):
        put_option = mt5.symbol_info(symbol)
        return put_option
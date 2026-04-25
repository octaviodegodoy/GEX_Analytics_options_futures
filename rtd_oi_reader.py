# -*- coding: utf-8 -*-
"""
Profit Pro RTD — Real-time OI via Windows COM RTD (with CSV fallback)
---------------------------------------------------------------------
Connects directly to Nelogica Profit Pro's RTD server to stream live
Open Interest data without requiring Excel or manual CSV exports.

Profit Pro RTD protocol (from Nelogica docs):
    ProgID : RTDTrading.RTDServer
    Topics : ["TICKER_SUFFIX", "ATTRIBUTE"]
    Suffix : B_0 (Bovespa), F_0 (BM&F)
    CAB    : Contratos Abertos (Open Interest)
    PEX    : Strike (Preço de Exercício)
    ULT    : Último (Last price)

Usage:
    from rtd_oi_reader import read_rtd_oi, rtd_data_changed

    # With tickers (from COTAHIST) — uses direct COM RTD
    df = read_rtd_oi(tickers=['BOVAT196', 'BOVAP196'], spot=195.0)

    # Without tickers — falls back to CSV
    df = read_rtd_oi(spot=195.0)

    # Check for data changes (works for both COM and CSV modes)
    if rtd_data_changed():
        ...  # recalculate GEX
"""
import os
import time
import pandas as pd
import re
import unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MQL5_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Default CSV fallback path
RTD_OI_PATH = os.path.join(MQL5_ROOT, 'Files', 'RTD_OI.csv')

# Profit Pro RTD COM settings
PROFIT_RTD_PROGID = "RTDTrading.RTDServer"
PROFIT_RTD_SUFFIX = "B_0"  # Bovespa

# Column name aliases for CSV fallback — Profit Pro uses Portuguese headers
_TICKER_ALIASES = {
    'codigo', 'codneg', 'ticker', 'symbol', 'asset', 'ativo', 'serie',
    'cod', 'instrumento', 'opcao',
}
_OI_ALIASES = {
    'qtd.aberta', 'qtd_aberta', 'qtdaberta', 'posição', 'posicao',
    'contratos_abertos', 'open_interest', 'openinterest', 'oi',
    'pos.aberta', 'pos_aberta', 'posaberta', 'contratos',
    'cont.abertos', 'cont_abertos', 'contabertos', 'cab',
}
_STRIKE_ALIASES = {
    'strike', 'exercicio', 'preco_exercicio', 'preço_exercício',
    'preco', 'pe',
}
_TYPE_ALIASES = {
    'tipo', 'type', 'call_put', 'c/p', 'cp', 'natureza',
}


def _normalize_colname(name: str) -> str:
    """Normalize CSV headers across locale/punctuation variants."""
    text = str(name).strip().lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


# =====================================================================
#  Windows COM RTD Client
# =====================================================================

class ProfitRTDClient:
    """
    Direct COM client for Nelogica Profit Pro RTD server.

    Just keep Profit Pro open — no Excel, no CSV export needed.
    Subscribes to CAB (Contratos Abertos / Open Interest) for each
    option ticker and returns live values.
    """

    def __init__(self):
        self._server = None
        self._callback = None
        self._topics = {}       # topic_id -> (ticker, attribute)
        self._reverse = {}      # (ticker, attribute) -> topic_id
        self._values = {}       # (ticker, attribute) -> value
        self._next_id = 0
        self._connected = False
        self._last_update = 0.0
        self._last_refresh = 0.0
        self._use_polling = False

    # ---- lifecycle ----

    def connect(self) -> bool:
        """Initialize COM and start the RTD server."""
        if self._connected:
            return True
        try:
            import pythoncom
            import win32com.client
            import win32com.server.util

            pythoncom.CoInitialize()

            self._server = win32com.client.Dispatch(PROFIT_RTD_PROGID)
            self._callback = _RTDUpdateEvent(self)
            wrapped = win32com.server.util.wrap(self._callback)
            result = self._server.ServerStart(wrapped)

            if result == 1:
                self._connected = True
                print(f"[RTD COM] Connected to {PROFIT_RTD_PROGID}")
                return True
            else:
                print(f"[RTD COM] ServerStart returned {result} (expected 1)")
                self._server = None
                return False
        except ImportError:
            print("[RTD COM] pywin32 not installed — pip install pywin32")
            return False
        except Exception as e:
            print(f"[RTD COM] Callback method failed: {e}")
            print("[RTD COM] Attempting simplified polling mode...")
            try:
                # Workaround: Try simpler connection without callback wrapper
                self._server = win32com.client.Dispatch(PROFIT_RTD_PROGID)
                self._connected = True
                self._use_polling = True
                print(f"[RTD COM] Connected to {PROFIT_RTD_PROGID} (polling mode)")
                return True
            except Exception as e2:
                print(f"[RTD COM] Polling mode also failed: {e2}")
                self._server = None
                return False

    def disconnect(self):
        """Terminate the RTD server connection."""
        if not self._connected or self._server is None:
            return
        try:
            for tid in list(self._topics.keys()):
                try:
                    self._server.DisconnectData(tid)
                except Exception:
                    pass
            self._server.ServerTerminate()
        except Exception:
            pass
        self._topics.clear()
        self._reverse.clear()
        self._values.clear()
        self._connected = False
        self._server = None
        print("[RTD COM] Disconnected")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_update(self) -> float:
        return self._last_update

    # ---- subscriptions ----

    def subscribe_oi(self, tickers: list, suffix: str = PROFIT_RTD_SUFFIX):
        """Subscribe to OI (CAB) for a list of option tickers."""
        self.subscribe_attributes(tickers=tickers, attributes=['CAB'], suffix=suffix)

    def subscribe_attributes(self, tickers: list, attributes: list, suffix: str = PROFIT_RTD_SUFFIX):
        """Subscribe to one or more RTD attributes for each ticker."""
        if not self._connected:
            return
        attrs = [str(a).strip().upper() for a in (attributes or []) if str(a).strip()]
        if not attrs:
            return
        new_count = 0
        for ticker in tickers:
            tk = str(ticker).strip().upper()
            if not tk:
                continue
            for attr in attrs:
                key = (tk, attr)
                if key in self._reverse:
                    continue  # already subscribed
                self._next_id += 1
                tid = self._next_id
                topic_str = f"{tk}_{suffix}"
                try:
                    new_values = True
                    result = self._server.ConnectData(
                        tid, [topic_str, attr], new_values
                    )
                    self._topics[tid] = key
                    self._reverse[key] = tid
                    # ConnectData may return an initial value
                    if result is not None:
                        try:
                            self._values[key] = float(result)
                        except (ValueError, TypeError):
                            pass
                    new_count += 1
                except Exception as e:
                    print(f"[RTD COM] Subscribe error {tk}/{attr}: {e}")
                    if _is_fatal_com_error(e):
                        _disable_rtd_com("fatal COM exception during ConnectData")
                        return
        if new_count > 0:
            print(f"[RTD COM] Subscribed to {new_count} new RTD topics "
                  f"(total: {len(self._topics)})")

    def refresh(self) -> dict:
        """
        Poll the server for updated values.
        Returns dict of {ticker: oi_value} for topics that changed.
        """
        if not self._connected:
            return {}
        try:
            topic_count = 0
            data = self._server.RefreshData(topic_count)
            if data is None:
                return {}
            # data is a 2D variant array: data[0]=topic IDs, data[1]=values
            updated = {}
            if hasattr(data, '__len__') and len(data) >= 2:
                ids = data[0]
                vals = data[1]
                n = len(ids) if hasattr(ids, '__len__') else 0
                for i in range(n):
                    tid = int(ids[i]) if ids[i] is not None else None
                    if tid and tid in self._topics:
                        key = self._topics[tid]
                        try:
                            val = float(vals[i])
                            self._values[key] = val
                            updated[key[0]] = val  # ticker -> oi
                        except (ValueError, TypeError):
                            pass
                if updated:
                    self._last_update = time.time()
            self._last_refresh = time.time()
            return updated
        except Exception as e:
            print(f"[RTD COM] Refresh error: {e}")
            if _is_fatal_com_error(e):
                _disable_rtd_com("fatal COM exception during RefreshData")
            return {}

    def get_all_oi(self) -> dict:
        """Return {ticker: oi} for all subscribed tickers with OI > 0."""
        result = {}
        for (ticker, attr), val in self._values.items():
            if attr == "CAB":
                try:
                    v = float(val)
                    if v > 0:
                        result[ticker] = v
                except (ValueError, TypeError):
                    pass
        return result

    def get_snapshot(self, tickers: list, attributes: list) -> pd.DataFrame:
        """Return current RTD snapshot for tickers/attributes as a DataFrame."""
        attrs = [str(a).strip().upper() for a in (attributes or []) if str(a).strip()]
        if not tickers or not attrs:
            return pd.DataFrame()

        rows = []
        for tk in tickers:
            ticker = str(tk).strip().upper()
            if not ticker:
                continue
            row = {'ticker': ticker}
            for attr in attrs:
                key = (ticker, attr)
                val = self._values.get(key, float('nan'))
                try:
                    row[attr.lower()] = float(val)
                except (ValueError, TypeError):
                    row[attr.lower()] = float('nan')
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)


class _RTDUpdateEvent:
    """
    IRTDUpdateEvent COM callback implementation.
    Called by the RTD server when it has new data available.
    """
    _public_methods_ = ['UpdateNotify', 'Disconnect']
    _public_attrs_ = ['HeartbeatInterval']

    def __init__(self, client: ProfitRTDClient):
        self.HeartbeatInterval = -1
        self._client = client

    def UpdateNotify(self):
        """Server signals that fresh data is available for RefreshData()."""
        self._client._last_update = time.time()

    def Disconnect(self):
        pass


# ---- Module-level singleton ----

_rtd_client: ProfitRTDClient = None
_rtd_com_disabled = False


def _is_fatal_com_error(exc: Exception) -> bool:
    """Identify fatal RTD COM errors where continuing calls is unsafe."""
    msg = str(exc).lower()
    return (
        "access violation" in msg
        or "-2147418113" in msg
        or "catastrophic failure" in msg
    )


def _disable_rtd_com(reason: str = ""):
    """Disable RTD COM for the current process and force CSV fallback."""
    global _rtd_com_disabled, _rtd_client
    _rtd_com_disabled = True
    if _rtd_client is not None:
        try:
            _rtd_client.disconnect()
        except Exception:
            pass
        _rtd_client = None
    if reason:
        print(f"[RTD COM] Disabled for this run: {reason}")


def _get_rtd_client() -> ProfitRTDClient:
    """Get or create the singleton RTD COM client."""
    global _rtd_client
    if _rtd_com_disabled:
        return None
    if _rtd_client is None:
        client = ProfitRTDClient()
        if client.connect():
            _rtd_client = client
        else:
            return None
    return _rtd_client


def rtd_shutdown():
    """Cleanly disconnect the RTD client (call at program exit)."""
    global _rtd_client
    if _rtd_client is not None:
        _rtd_client.disconnect()
        _rtd_client = None


# =====================================================================
#  CSV Fallback Reader
# =====================================================================

#: Maximum age (seconds) for the RTD CSV to be considered current-day data.
#: RTD readings are only valid for the current trading day; a stale file from a
#: previous session must not silently feed the GEX calculation.
RTD_CSV_MAX_AGE_SECONDS = 30 * 60  # 30 minutes


def _is_csv_fresh(path: str, max_age_seconds: int = RTD_CSV_MAX_AGE_SECONDS) -> bool:
    """Return True if the CSV mtime is from today and within max_age_seconds."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return False
    import datetime as _dt
    now = time.time()
    age = now - mtime
    today = _dt.date.today()
    file_date = _dt.date.fromtimestamp(mtime)
    if file_date != today:
        print(f"[RTD CSV] Rejecting stale file: mtime date {file_date} != today {today} "
              f"({path})")
        return False
    if age > max_age_seconds:
        print(f"[RTD CSV] Rejecting stale file: age {age:.0f}s > "
              f"max {max_age_seconds}s ({path})")
        return False
    return True


def _read_csv_oi(filepath: str = None, spot: float = None,
                 strikes_around: int = 15,
                 enforce_freshness: bool = True,
                 max_age_seconds: int = RTD_CSV_MAX_AGE_SECONDS) -> pd.DataFrame:
    """Read OI from a Profit Pro CSV export (fallback when COM unavailable).

    By default, files older than ``max_age_seconds`` or dated before today are
    rejected to prevent stale OI from a previous session leaking into the GEX
    parameters. Pass ``enforce_freshness=False`` for backtest/inspection only.
    """
    path = filepath or RTD_OI_PATH

    if not os.path.exists(path):
        return pd.DataFrame()

    if enforce_freshness and not _is_csv_fresh(path, max_age_seconds=max_age_seconds):
        return pd.DataFrame()

    df = None
    for enc in ('utf-8-sig', 'latin-1', 'cp1252'):
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine='python')
            break
        except Exception:
            continue

    if df is None or df.empty:
        print(f"[RTD CSV] Could not read {path}")
        return pd.DataFrame()

    # --- Normalise column names ---
    col_map = {}
    for c in df.columns:
        cl = _normalize_colname(c)
        if cl in _TICKER_ALIASES:
            col_map[c] = 'ticker'
        elif cl in _OI_ALIASES:
            col_map[c] = 'oi'
        elif cl in _STRIKE_ALIASES:
            col_map[c] = 'strike'
        elif cl in _TYPE_ALIASES:
            col_map[c] = 'type'

    df = df.rename(columns=col_map)

    if 'ticker' not in df.columns or 'oi' not in df.columns:
        print(f"[RTD CSV] Missing required columns. Found: {list(df.columns)}")
        print(f"[RTD CSV] Need 'ticker' + 'oi' (or Profit Pro equivalents)")
        return pd.DataFrame()

    # Clean up
    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
    df['oi'] = pd.to_numeric(df['oi'], errors='coerce').fillna(0)
    df = df[df['oi'] > 0].copy()

    # --- Filter to ±strikes_around strikes around spot ---
    if spot is not None and spot > 0 and 'strike' in df.columns:
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df = df.dropna(subset=['strike'])
        unique_strikes = sorted(df['strike'].unique())
        if unique_strikes:
            import bisect
            idx = bisect.bisect_left(unique_strikes, spot)
            lo = max(0, idx - strikes_around)
            hi = min(len(unique_strikes), idx + strikes_around)
            keep = set(unique_strikes[lo:hi])
            before = len(df)
            df = df[df['strike'].isin(keep)]
            print(f"[RTD CSV] Filtered to {len(keep)} strikes around spot "
                f"{spot:.2f} ({before} -> {len(df)} rows)")

    return df[['ticker', 'oi']]


# =====================================================================
#  Public API
# =====================================================================

def read_rtd_oi(filepath: str = None, spot: float = None,
                strikes_around: int = 15,
                tickers: list = None,
                enforce_freshness: bool = True,
                max_age_seconds: int = RTD_CSV_MAX_AGE_SECONDS) -> pd.DataFrame:
    """
    Get real-time OI data from Profit Pro.

    Strategy:
      1. If ``tickers`` provided → try COM RTD server (live, no export needed)
      2. Fall back to CSV file (manual or auto-exported)

    Parameters
    ----------
    filepath : str, optional
        Path to CSV fallback file.
    spot : float, optional
        Current spot price for strike filtering (CSV mode only).
    strikes_around : int
        Strikes to keep on each side of spot (default 15, CSV mode only).
    tickers : list of str, optional
        Option ticker codes (e.g. ['BOVAT196', 'BOVAP196']).
        When provided, enables direct COM RTD connection to Profit Pro.
    enforce_freshness : bool
        Reject CSV files dated before today or older than ``max_age_seconds``.
        RTD data is only valid for the current trading day.
    max_age_seconds : int
        Maximum CSV file age accepted when ``enforce_freshness`` is True.

    Returns
    -------
    pd.DataFrame
        Columns: ticker (str), oi (float).
    """
    # --- Strategy 1: Direct COM RTD (preferred when tickers known) ---
    if tickers:
        client = _get_rtd_client()
        if client is not None:
            try:
                client.subscribe_oi(tickers)
                client.refresh()
                oi_map = client.get_all_oi()
                if oi_map:
                    df = pd.DataFrame([
                        {'ticker': t, 'oi': v}
                        for t, v in oi_map.items()
                    ])
                    if not df.empty:
                        print(f"[RTD COM] Live OI: {len(df)} options with OI > 0")
                        return df
                print("[RTD COM] No OI values yet (se,
                        enforce_freshness=enforce_freshness,
                        max_age_seconds=max_age_secondsrver warming up?) "
                      "— trying CSV fallback")
            except Exception as e:
                print(f"[RTD COM] Error: {e} — trying CSV fallback")

    # --- Strategy 2: CSV fallback ---
    return _read_csv_oi(filepath=filepath, spot=spot,
                        strikes_around=strikes_around)


def read_rtd_option_snapshot(tickers: list,
                             attributes: list = None,
                             wait_seconds: float = 1.0,
                             refresh_rounds: int = 3) -> pd.DataFrame:
    """
    Read RTD snapshot for option tickers and attributes (e.g., ULT/PEX/CAB).

    Returns
    -------
    pd.DataFrame
        Columns: ticker plus one column per requested attribute (lowercase).
        Example: ticker, ult, pex, cab
    """
    if not tickers:
        return pd.DataFrame()

    attrs = [str(a).strip().upper() for a in (attributes or ['CAB']) if str(a).strip()]
    if not attrs:
        attrs = ['CAB']

    client = _get_rtd_client()
    if client is None:
        # Graceful fallback: if only CAB requested, map CSV/COM OI reader output.
        if attrs == ['CAB']:
            oi_df = read_rtd_oi(tickers=tickers)
            if oi_df is not None and not oi_df.empty:
                out = oi_df.copy()
                out = out.rename(columns={'oi': 'cab'})
                if 'ticker' in out.columns:
                    out['ticker'] = out['ticker'].astype(str).str.strip().str.upper()
                return out[['ticker', 'cab']]
        return pd.DataFrame()

    try:
        client.subscribe_attributes(tickers=tickers, attributes=attrs)
        rounds = max(int(refresh_rounds), 1)
        per_round_wait = max(float(wait_seconds), 0.0) / float(rounds)
        for i in range(rounds):
            client.refresh()
            if per_round_wait > 0 and i < rounds - 1:
                time.sleep(per_round_wait)
        return client.get_snapshot(tickers=tickers, attributes=attrs)
    except Exception as e:
        print(f"[RTD COM] Snapshot error: {e}")
        return pd.DataFrame()


def rtd_data_changed() -> bool:
    """
    Check whether RTD data has been updated since the last check.

    For COM mode: checks if the RTD server signalled new data.
    For CSV mode: checks if the file modification time changed.

    Returns True if fresh data is available.
    """
    global _rtd_last_seen

    # COM mode — actively refresh and check
    if _rtd_client is not None and _rtd_client.connected:
        ts = _rtd_client.last_update
        if ts > _rtd_last_seen:
            _rtd_last_seen = ts
            return True
        # Proactively poll for any pending data
        updated = _rtd_client.refresh()
        if updated:
            _rtd_last_seen = _rtd_client.last_update
            return True
        return False

    # CSV fallback — check file mtime
    mtime = rtd_file_mtime()
    if mtime > _rtd_last_seen:
        _rtd_last_seen = mtime
        return True
    return False


_rtd_last_seen = 0.0


def rtd_file_mtime(filepath: str = None) -> float:
    """Return the CSV file modification timestamp, or 0.0 if not found."""
    path = filepath or RTD_OI_PATH
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

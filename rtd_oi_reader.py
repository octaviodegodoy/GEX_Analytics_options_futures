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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MQL5_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Default CSV fallback path
RTD_OI_PATH = os.path.join(MQL5_ROOT, 'Files', 'RTD_OI.csv')

# Profit Pro RTD COM settings
PROFIT_RTD_PROGID = "RTDTrading.RTDServer"
PROFIT_RTD_SUFFIX = "B_0"  # Bovespa

# Column name aliases for CSV fallback — Profit Pro uses Portuguese headers
_TICKER_ALIASES = {
    'codigo', 'codneg', 'ticker', 'symbol', 'ativo', 'serie',
    'cod', 'instrumento', 'opcao',
}
_OI_ALIASES = {
    'qtd.aberta', 'qtd_aberta', 'qtdaberta', 'posição', 'posicao',
    'contratos_abertos', 'open_interest', 'openinterest', 'oi',
    'pos.aberta', 'pos_aberta', 'posaberta', 'contratos',
}
_STRIKE_ALIASES = {
    'strike', 'exercicio', 'preco_exercicio', 'preço_exercício',
    'preco', 'pe',
}
_TYPE_ALIASES = {
    'tipo', 'type', 'call_put', 'c/p', 'cp', 'natureza',
}


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
            print(f"[RTD COM] Connection failed: {e}")
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
        if not self._connected:
            return
        new_count = 0
        for ticker in tickers:
            key = (ticker, "CAB")
            if key in self._reverse:
                continue  # already subscribed
            self._next_id += 1
            tid = self._next_id
            topic_str = f"{ticker}_{suffix}"
            try:
                new_values = True
                result = self._server.ConnectData(
                    tid, [topic_str, "CAB"], new_values
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
                print(f"[RTD COM] Subscribe error {ticker}: {e}")
        if new_count > 0:
            print(f"[RTD COM] Subscribed to {new_count} new OI topics "
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


def _get_rtd_client() -> ProfitRTDClient:
    """Get or create the singleton RTD COM client."""
    global _rtd_client
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

def _read_csv_oi(filepath: str = None, spot: float = None,
                 strikes_around: int = 15) -> pd.DataFrame:
    """Read OI from a Profit Pro CSV export (fallback when COM unavailable)."""
    path = filepath or RTD_OI_PATH

    if not os.path.exists(path):
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
        cl = c.lower().strip().replace(' ', '_')
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
                  f"{spot:.2f} ({before} → {len(df)} rows)")

    return df[['ticker', 'oi']]


# =====================================================================
#  Public API
# =====================================================================

def read_rtd_oi(filepath: str = None, spot: float = None,
                strikes_around: int = 15,
                tickers: list = None) -> pd.DataFrame:
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
                print("[RTD COM] No OI values yet (server warming up?) "
                      "— trying CSV fallback")
            except Exception as e:
                print(f"[RTD COM] Error: {e} — trying CSV fallback")

    # --- Strategy 2: CSV fallback ---
    return _read_csv_oi(filepath=filepath, spot=spot,
                        strikes_around=strikes_around)


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

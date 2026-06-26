"""
XAUUSD_Live_Bot.py
==================
Paper trading bot for XAUUSD M15 MSB strategy.

FROZEN SYSTEM:
- Long-only | 4H Bull Trend | Bullish MSB | Immediate Entry
- Ensemble Score = 0.20 * Expansion + 0.40 * TrendCont - 0.40 * FakeBreakout
- Threshold (normalized): >= 0.65
- RR = 1.5
- No training inside this file. Load frozen model files only.

Execution modes:
  PAPER_INTERNAL    - prints decisions only, no MT5 orders
  PAPER_MT5_DEMO    - sends live demo orders to MT5 (paper money)
  LIVE_DISABLED     - live trading blocked (requires explicit override)

Telegram alerts: required - set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in config.
"""

import os
import sys
import time
import pickle
import logging
import warnings
import datetime
import traceback
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# FROZEN STRATEGY CONSTANTS  (DO NOT CHANGE)
# -----------------------------------------------------------------------------
W_EXP       = 0.20
W_CONT      = 0.40
W_FAKE      = 0.40
THRESHOLD   = 0.65      # normalized ensemble score
RR          = 1.5
DIRECTION   = "LONG"    # long-only
SYMBOL      = "XAUUSD"
TF_M15      = None      # set after mt5 import
TF_H4       = None      # set after mt5 import

# -----------------------------------------------------------------------------
# CONFIGURATION  (edit here before running)
# -----------------------------------------------------------------------------
EXECUTION_MODE = "PAPER_MT5_DEMO"   # PAPER_INTERNAL | PAPER_MT5_DEMO | LIVE_DISABLED

# Risk
MAX_RISK_PER_TRADE    = 0.005   # 0.50% of balance
MAX_LOT               = 0.20    # safety cap for demo
MIN_LOT               = None    # read from broker on startup
LOT_STEP              = None    # read from broker on startup
MAX_OPEN_POSITIONS    = 1

# Daily / Overall risk limits
DAILY_LOSS_LIMIT_PCT      = 2.0   # halt if daily PnL <= -2%
MAX_TOTAL_DRAWDOWN_PCT    = 6.0   # halt if total DD <= -6%
STOP_AFTER_CONSEC_LOSSES  = 2     # halt day after N consecutive losses
MAX_TRADES_PER_DAY        = 2

# Candle lookback for live data
M15_BARS_NEEDED  = 2000   # enough for indicators + 4H trend mapping
H4_BARS_NEEDED   = 500

# Polling
POLL_INTERVAL_SEC = 60    # check MT5 every 60 seconds

# Telegram (set your credentials here)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATHS = {
    "expansion":    os.path.join(MODEL_DIR, "expansion_model.pkl"),
    "fake":         os.path.join(MODEL_DIR, "fake_breakout_model.pkl"),
    "trend":        os.path.join(MODEL_DIR, "trend_continuation_model.pkl"),
    "metadata":     os.path.join(MODEL_DIR, "feature_metadata.pkl"),
}

# Log files
DECISION_LOG = os.path.join(os.path.dirname(__file__), "live_decision_log.csv")
TRADE_LOG    = os.path.join(os.path.dirname(__file__), "live_trade_log.csv")

# State file (restart-safe candle tracking)
STATE_FILE   = os.path.join(os.path.dirname(__file__), "state.json")

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
_fh = logging.FileHandler(
    os.path.join(os.path.dirname(__file__), "live_bot.log"),
    encoding="utf-8"
)
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])
log = logging.getLogger("LiveBot")

# -----------------------------------------------------------------------------
# STATE FILE HELPERS  (restart-safe candle tracking)
# -----------------------------------------------------------------------------
import json

def load_state() -> dict:
    """Load state.json from disk. Returns empty dict if file does not exist."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"[STATE] Could not read state.json: {e}")
    return {}

def save_state(state: dict):
    """Atomically write state dict to state.json."""
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        log.warning(f"[STATE] Could not write state.json: {e}")

# -----------------------------------------------------------------------------
# TELEGRAM HELPER
# -----------------------------------------------------------------------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def alert(msg: str):
    log.info(msg)
    send_telegram(msg)


# -----------------------------------------------------------------------------
# CSV LOGGING HELPERS
# -----------------------------------------------------------------------------
DECISION_COLS = [
    "time", "symbol", "m15_candle_time", "signal", "trend_4h", "msb_detected",
    "expansion_prob", "fake_breakout_prob", "trend_prob",
    "ensemble_score", "ensemble_score_norm",
    "decision", "reason", "entry", "sl", "tp", "lot", "risk_pct",
]
TRADE_COLS = [
    "open_time", "close_time", "symbol", "direction", "entry", "sl", "tp",
    "lot", "result", "pnl_usd", "ticket",
]

def _ensure_csv(path: str, cols: list):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

def log_decision(row: dict):
    _ensure_csv(DECISION_LOG, DECISION_COLS)
    with open(DECISION_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DECISION_COLS)
        w.writerow({k: row.get(k, "") for k in DECISION_COLS})

def log_trade(row: dict):
    _ensure_csv(TRADE_LOG, TRADE_COLS)
    with open(TRADE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_COLS)
        w.writerow({k: row.get(k, "") for k in TRADE_COLS})


# -----------------------------------------------------------------------------
# MT5 CONNECTION
# -----------------------------------------------------------------------------
def init_mt5():
    global TF_M15, TF_H4
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        sys.exit(1)

    TF_M15 = mt5.TIMEFRAME_M15
    TF_H4  = mt5.TIMEFRAME_H4

    if not mt5.initialize():
        err = mt5.last_error()
        alert(f"[BOT] MT5 init failed: {err}")
        sys.exit(1)

    # Ensure XAUUSD is in Market Watch
    if not mt5.symbol_select(SYMBOL, True):
        alert(f"[BOT] Could not select {SYMBOL} in Market Watch")
        mt5.shutdown()
        sys.exit(1)

    alert("[BOT] MT5 connected successfully.")
    return mt5


# -----------------------------------------------------------------------------
# LIVE DATA LOADING  (replaces pd.read_csv)
# -----------------------------------------------------------------------------
def fetch_bars(mt5, symbol: str, timeframe, n_bars: int, label: str) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch {label} bars: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["Date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "tick_volume": "Volume"}, inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# -----------------------------------------------------------------------------
# BROKER INFO
# -----------------------------------------------------------------------------
def get_broker_lot_info(mt5) -> tuple:
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return 0.01, 0.01
    return info.volume_min, info.volume_step


def get_account_info(mt5) -> dict:
    acc = mt5.account_info()
    if acc is None:
        return {}
    return {
        "login":   acc.login,
        "server":  acc.server,
        "balance": acc.balance,
        "equity":  acc.equity,
        "currency": acc.currency,
    }


# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
def load_models() -> tuple:
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            alert(f"[BOT] Model file missing: {path}\nRun train_models.py first.")
            sys.exit(1)

    with open(MODEL_PATHS["expansion"], "rb") as f:
        model_a = pickle.load(f)
    with open(MODEL_PATHS["fake"], "rb") as f:
        model_c = pickle.load(f)
    with open(MODEL_PATHS["trend"], "rb") as f:
        model_d = pickle.load(f)
    with open(MODEL_PATHS["metadata"], "rb") as f:
        meta = pickle.load(f)

    log.info(f"  Models loaded. Features: {meta['n_features']}")
    log.info(f"  Trained on: {meta['date_range']['start']} -> {meta['date_range']['end']}")
    return model_a, model_c, model_d, meta


# -----------------------------------------------------------------------------
# INDICATOR PIPELINE  (reused identically from backtest)
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from XAUUSD_Colab_Backtest import (
    CONFIG,
    add_m15_indicators,
    build_4h_trend,
    merge_4h_into_m15,
    detect_swings_no_repaint,
    detect_msb,
    generate_signals,
    compute_sl_tp,
)


def build_live_df(df_m15: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """Apply identical pipeline as backtest: indicators -> swings -> MSB -> signals."""
    df4h_trend = build_4h_trend(df_4h)
    df = merge_4h_into_m15(df_m15, df4h_trend)
    df = add_m15_indicators(df)

    high = df["High"].values
    low  = df["Low"].values
    last_sw_high, last_sw_low, sw_high_idx, sw_low_idx = detect_swings_no_repaint(
        high, low, CONFIG["swing_length"]
    )
    df["last_sw_high"] = last_sw_high
    df["last_sw_low"]  = last_sw_low
    df["sw_high_idx"]  = sw_high_idx
    df["sw_low_idx"]   = sw_low_idx

    bull_msb, bear_msb, msb_size = detect_msb(
        df["Close"].values, last_sw_high, last_sw_low
    )
    df["bull_msb"] = bull_msb
    df["bear_msb"] = bear_msb
    df["msb_size"] = msb_size

    # Long-only: allow longs, disable shorts
    df = generate_signals(df, allow_longs=True, allow_shorts=False)
    return df


# -----------------------------------------------------------------------------
# LIVE FEATURE EXTRACTION  (single candle at signal bar)
# -----------------------------------------------------------------------------
def extract_live_features(df: pd.DataFrame, sig_i: int, meta: dict) -> np.ndarray:
    """
    Extract the exact same V2 feature vector for a single bar (sig_i).
    Replicates extract_ml_features_v2 logic but for one bar.
    Returns array of shape (1, n_features) for model.predict_proba.
    """
    feature_cols = meta["feature_cols"]
    col_medians  = meta["col_medians"]

    C   = df["Close"]
    O   = df["Open"]
    H   = df["High"]
    L   = df["Low"]
    ATR = df["atr14"].replace(0, np.nan)

    row = {}

    # -- GROUP 1: LIQUIDITY (LIQ_) --
    new_day = df["Date"].dt.date != df["Date"].shift(1).dt.date
    day_id  = new_day.cumsum()
    pdh = day_id.map(df.groupby(day_id)["High"].max().shift(1)).ffill()
    pdl = day_id.map(df.groupby(day_id)["Low"].min().shift(1)).ffill()
    row["LIQ_Dist_PDH"] = (C.iloc[sig_i] - pdh.iloc[sig_i]) / ATR.iloc[sig_i]
    row["LIQ_Dist_PDL"] = (C.iloc[sig_i] - pdl.iloc[sig_i]) / ATR.iloc[sig_i]

    hour    = df["Date"].dt.hour
    is_asia = ((hour >= 21) | (hour < 7)).astype(int)
    asia_start = (hour == 21) & (hour.shift(1) != 21)
    asia_id = asia_start.cumsum()
    asia_h  = df.groupby(asia_id)["High"].cummax().ffill()
    asia_l  = df.groupby(asia_id)["Low"].cummin().ffill()
    row["LIQ_Dist_Asia_H"] = (C.iloc[sig_i] - asia_h.iloc[sig_i]) / ATR.iloc[sig_i]
    row["LIQ_Dist_Asia_L"] = (C.iloc[sig_i] - asia_l.iloc[sig_i]) / ATR.iloc[sig_i]

    row["LIQ_Dist_5D_H"]  = (C.iloc[sig_i] - H.rolling(5*96).max().shift(1).iloc[sig_i]) / ATR.iloc[sig_i]
    row["LIQ_Dist_5D_L"]  = (C.iloc[sig_i] - L.rolling(5*96).min().shift(1).iloc[sig_i]) / ATR.iloc[sig_i]
    row["LIQ_Dist_20D_H"] = (C.iloc[sig_i] - H.rolling(20*96).max().shift(1).iloc[sig_i]) / ATR.iloc[sig_i]
    row["LIQ_Dist_20D_L"] = (C.iloc[sig_i] - L.rolling(20*96).min().shift(1).iloc[sig_i]) / ATR.iloc[sig_i]

    pdh_swept = ((H > pdh) & (C < pdh)).astype(int)
    pdl_swept = ((L < pdl) & (C > pdl)).astype(int)
    row["LIQ_PDH_Swept"] = pdh_swept.iloc[sig_i]
    row["LIQ_PDL_Swept"] = pdl_swept.iloc[sig_i]

    min_bars = min(
        (pdh_swept == 1).groupby((pdh_swept == 1).cumsum()).cumcount().iloc[sig_i],
        (pdl_swept == 1).groupby((pdl_swept == 1).cumsum()).cumcount().iloc[sig_i],
    )
    row["LIQ_Bars_Since_Sweep"] = min_bars

    # -- GROUP 2: STRUCTURE MEMORY (MEM_) --
    trend = df["4h_trend"].ffill().fillna(0)
    row["MEM_Trend_Age"] = trend.groupby((trend != trend.shift()).cumsum()).cumcount().iloc[sig_i]
    row["MEM_Trend_Dir"] = trend.iloc[sig_i]

    bull_msb_s = df.get("bull_msb", pd.Series(0, index=df.index))
    bear_msb_s = df.get("bear_msb", pd.Series(0, index=df.index))
    msb_event  = (bull_msb_s | bear_msb_s).astype(int)
    row["MEM_Bars_Since_MSB"]    = msb_event.groupby(msb_event.cumsum()).cumcount().iloc[sig_i]
    row["MEM_MSB_Count_Session"] = msb_event.groupby(day_id).cumsum().iloc[sig_i]

    last_sw_h  = df.get("last_sw_high", pd.Series(np.nan, index=df.index))
    last_sw_l  = df.get("last_sw_low",  pd.Series(np.nan, index=df.index))
    swing_size = (last_sw_h - last_sw_l) / ATR
    row["MEM_Swing_Size_ATR"]  = swing_size.iloc[sig_i]
    row["MEM_Swing_Expansion"] = (swing_size / swing_size.shift(1).replace(0, np.nan)).iloc[sig_i]
    row["MEM_HH_Count"] = (last_sw_h > last_sw_h.shift(1)).groupby(
        (last_sw_h < last_sw_h.shift(1)).cumsum()).cumcount().iloc[sig_i]
    row["MEM_LL_Count"] = (last_sw_l < last_sw_l.shift(1)).groupby(
        (last_sw_l > last_sw_l.shift(1)).cumsum()).cumcount().iloc[sig_i]

    # -- GROUP 3: COMPRESSION (VOL_) --
    atr_100 = df["High"].rolling(100).max() - df["Low"].rolling(100).min()
    row["VOL_ATR_Compression"] = (ATR / atr_100.replace(0, np.nan)).iloc[sig_i]
    row["VOL_ATR_ZScore"]      = ((ATR - ATR.rolling(100).mean()) / ATR.rolling(100).std().replace(0, np.nan)).iloc[sig_i]
    row["VOL_Roll_Std"]        = (C.rolling(20).std() / C.rolling(20).mean()).iloc[sig_i]
    row["VOL_Roll_CV"]         = row["VOL_Roll_Std"] * 100

    bb_w = (C.rolling(20).std() * 4) / C.rolling(20).mean()
    row["VOL_BB_Width_Pct"] = bb_w.rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
    ).iloc[sig_i]

    tr = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs(),
    ], axis=1).max(axis=1)
    row["VOL_Is_NR4"]      = int(tr.iloc[sig_i] < tr.rolling(4).min().shift(1).iloc[sig_i])
    row["VOL_Is_NR7"]      = int(tr.iloc[sig_i] < tr.rolling(7).min().shift(1).iloc[sig_i])
    row["VOL_Inside_Bar"]  = int(H.iloc[sig_i] < H.iloc[sig_i-1] and L.iloc[sig_i] > L.iloc[sig_i-1])
    inside_bar_s           = ((H < H.shift(1)) & (L > L.shift(1))).astype(int)
    row["VOL_Inside_Streak"] = inside_bar_s.groupby((inside_bar_s == 0).cumsum()).cumcount().iloc[sig_i]

    # -- GROUP 4: GEOMETRY (GEO_) --
    hl_range = (H - L).replace(0, np.nan)
    row["GEO_Body_Pct"]     = ((C - O).abs() / hl_range).iloc[sig_i]
    row["GEO_Up_Wick_Pct"]  = ((H - df[["Open","Close"]].max(axis=1)) / hl_range).iloc[sig_i]
    row["GEO_Dn_Wick_Pct"]  = ((df[["Open","Close"]].min(axis=1) - L) / hl_range).iloc[sig_i]
    row["GEO_Retracement_3B"]  = (df["High"].rolling(3).max() - df["Low"].rolling(3).min()).iloc[sig_i] / ATR.iloc[sig_i]
    row["GEO_Consecutive_Bull"] = (C > O).astype(int).groupby((C <= O).cumsum()).cumcount().iloc[sig_i]
    row["GEO_Consecutive_Bear"] = (C < O).astype(int).groupby((C >= O).cumsum()).cumcount().iloc[sig_i]
    row["GEO_Momentum_Score"]   = ((C - C.shift(5)) / ATR).iloc[sig_i]

    sw_h_val = last_sw_h.iloc[sig_i]
    sw_l_val = last_sw_l.iloc[sig_i]
    c_price  = C.iloc[sig_i]
    atr_val  = ATR.iloc[sig_i] if not pd.isna(ATR.iloc[sig_i]) else 1.0

    row["GEO_MSB_Dist_Swing_H"] = (sw_h_val - c_price) / atr_val if not pd.isna(sw_h_val) else 0
    row["GEO_MSB_Dist_Swing_L"] = (c_price - sw_l_val) / atr_val if not pd.isna(sw_l_val) else 0
    sw_size_val = (sw_h_val - sw_l_val) / atr_val if (not pd.isna(sw_h_val) and not pd.isna(sw_l_val)) else 1
    row["GEO_Breakout_Size"]    = (c_price - sw_h_val) / atr_val if not pd.isna(sw_h_val) else 0
    row["GEO_Breakout_vs_Swing"] = row["GEO_Breakout_Size"] / sw_size_val if sw_size_val > 0 else 0

    # Build feature vector in exactly the same column order as training
    feat_vec = []
    for col in feature_cols:
        val = row.get(col, col_medians.get(col, 0.0))
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = col_medians.get(col, 0.0)
        feat_vec.append(float(val))

    return np.array(feat_vec).reshape(1, -1)


# -----------------------------------------------------------------------------
# LOT SIZE CALCULATION
# -----------------------------------------------------------------------------
def calc_lot_size(balance: float, entry: float, sl: float,
                  min_lot: float, lot_step: float,
                  contract_size: float = 100.0) -> tuple:
    """
    Calculate lot size so that loss at SL = MAX_RISK_PER_TRADE * balance.

    For XAUUSD: 1 standard lot = 100 oz
    PnL = (exit - entry) * lot_size * contract_size
    risk_amount = (entry - sl) * lot_size * contract_size
    lot_size = risk_amount / ((entry - sl) * contract_size)
    """
    risk_amount  = balance * MAX_RISK_PER_TRADE
    stop_dist    = abs(entry - sl)
    if stop_dist <= 0:
        return None, "SL distance is zero"

    raw_lot = risk_amount / (stop_dist * contract_size)

    # Round down to lot_step precision
    lot = max(min_lot, round(raw_lot // lot_step * lot_step, 8))

    note = ""
    if lot < min_lot:
        return None, f"SKIPPED: lot {lot:.4f} below minimum {min_lot}"
    if lot > MAX_LOT:
        lot  = MAX_LOT
        note = "LOT CAPPED"

    return lot, note


# -----------------------------------------------------------------------------
# ORDER EXECUTION
# -----------------------------------------------------------------------------
def place_order(mt5, entry: float, sl: float, tp: float, lot: float) -> dict:
    if EXECUTION_MODE == "PAPER_INTERNAL":
        log.info(f"[PAPER_INTERNAL] LONG {lot} lots @ {entry:.2f} | SL {sl:.2f} | TP {tp:.2f}")
        return {"retcode": 0, "order": 99999}

    if EXECUTION_MODE == "LIVE_DISABLED":
        alert("[BOT] Live trading is disabled. EXECUTION_MODE=LIVE_DISABLED")
        return None

    # PAPER_MT5_DEMO: real demo order
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        log.error("Could not get tick data.")
        return None

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    SYMBOL,
        "volume":    float(lot),
        "type":      mt5.ORDER_TYPE_BUY,
        "price":     tick.ask,
        "sl":        round(sl, 2),
        "tp":        round(tp, 2),
        "deviation": 20,
        "magic":     20240001,
        "comment":   "XAUUSD_LiveBot_MSB",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error(f"order_send returned None: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed. retcode={result.retcode} | comment={result.comment}")
        return None

    return {"retcode": result.retcode, "order": result.order}


# -----------------------------------------------------------------------------
# OPEN POSITION CHECK
# -----------------------------------------------------------------------------
def count_open_positions(mt5) -> int:
    if EXECUTION_MODE == "PAPER_INTERNAL":
        return 0
    positions = mt5.positions_get(symbol=SYMBOL)
    return len(positions) if positions else 0


# -----------------------------------------------------------------------------
# RISK STATE  (persists across candle evaluations each trading day)
# -----------------------------------------------------------------------------
class RiskState:
    def __init__(self):
        self.reset_day()
        self.total_equity_start = None
        self.total_drawdown_pct = 0.0

    def reset_day(self):
        """Called at Midnight UTC daily reset."""
        self.daily_pnl_pct     = 0.0
        self.trades_today      = 0
        self.consec_losses     = 0
        self.halted_today      = False
        self.last_reset_date   = datetime.datetime.utcnow().date()

    def check_daily_reset(self):
        today = datetime.datetime.utcnow().date()
        if today != self.last_reset_date:
            log.info("[RISK] Midnight UTC - resetting daily limits.")
            self.reset_day()

    def can_trade(self) -> tuple:
        """Returns (ok: bool, reason: str)."""
        self.check_daily_reset()

        if self.halted_today:
            return False, "Daily halt active"
        if self.daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
            self.halted_today = True
            return False, f"Daily loss limit hit ({self.daily_pnl_pct:.2f}%)"
        if self.total_drawdown_pct <= -MAX_TOTAL_DRAWDOWN_PCT:
            return False, f"Max total drawdown hit ({self.total_drawdown_pct:.2f}%)"
        if self.consec_losses >= STOP_AFTER_CONSEC_LOSSES:
            self.halted_today = True
            return False, f"{STOP_AFTER_CONSEC_LOSSES} consecutive losses - halted for today"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Max trades/day reached ({MAX_TRADES_PER_DAY})"

        return True, "OK"

    def record_trade(self, pnl_pct: float):
        self.daily_pnl_pct    += pnl_pct
        self.total_drawdown_pct = min(self.total_drawdown_pct, self.daily_pnl_pct)
        self.trades_today      += 1
        if pnl_pct < 0:
            self.consec_losses += 1
        else:
            self.consec_losses  = 0


# -----------------------------------------------------------------------------
# MAIN EVALUATION LOOP
# -----------------------------------------------------------------------------
class LiveBot:
    def __init__(self):
        self.mt5               = None
        self.model_a           = None
        self.model_c           = None
        self.model_d           = None
        self.meta              = None
        self.last_m15_time     = None
        self.risk              = RiskState()
        self.min_lot           = 0.01
        self.lot_step          = 0.01

        # Load persisted candle time from previous run (restart-safe)
        state = load_state()
        saved = state.get("last_processed_m15_candle_time")
        if saved:
            try:
                self.last_m15_time = pd.Timestamp(saved)
                log.info(f"[STATE] Resumed. Last processed candle: {self.last_m15_time}")
            except Exception:
                log.warning("[STATE] Could not parse saved candle time. Starting fresh.")

    def startup(self):
        log.info("=" * 60)
        log.info("  XAUUSD LIVE BOT STARTING")
        log.info(f"  Execution Mode : {EXECUTION_MODE}")
        log.info(f"  Symbol         : {SYMBOL}")
        log.info(f"  Frozen Strategy: Long-only | 4H Bull | MSB | Ensemble >= {THRESHOLD}")
        log.info(f"  Risk/Trade     : {MAX_RISK_PER_TRADE*100:.2f}% | MaxLot: {MAX_LOT}")
        log.info("=" * 60)

        # Connect MT5
        self.mt5 = init_mt5()

        # Account info
        acc = get_account_info(self.mt5)
        self.risk.total_equity_start = acc.get("balance", 10000)
        self.min_lot, self.lot_step  = get_broker_lot_info(self.mt5)

        alert(
            f"[BOT STARTED]\n"
            f"Account : {acc.get('login')}\n"
            f"Server  : {acc.get('server')}\n"
            f"Balance : {acc.get('balance')} {acc.get('currency')}\n"
            f"Symbol  : {SYMBOL}\n"
            f"Mode    : {EXECUTION_MODE}\n"
            f"Risk    : {MAX_RISK_PER_TRADE*100:.2f}% | MaxLot: {MAX_LOT}"
        )

        # Load frozen models
        self.model_a, self.model_c, self.model_d, self.meta = load_models()
        alert(
            f"[BOT] Models loaded\n"
            f"Features: {self.meta['n_features']}\n"
            f"Trained : {self.meta['date_range']['start']} -> {self.meta['date_range']['end']}"
        )

        open_pos = count_open_positions(self.mt5)
        log.info(f"  Open positions : {open_pos}")

    def evaluate_candle(self, df: pd.DataFrame, sig_i: int) -> dict:
        """
        Evaluate one closed M15 candle. Returns decision dict.
        """
        bar_time  = df["Date"].iloc[sig_i]
        signal    = df["signal"].iloc[sig_i]
        trend_4h  = df["4h_trend"].iloc[sig_i]
        msb_det   = bool(df["bull_msb"].iloc[sig_i])

        base_rec = {
            "time":          datetime.datetime.utcnow().isoformat(),
            "symbol":        SYMBOL,
            "m15_candle_time": str(bar_time),
            "signal":        int(signal),
            "trend_4h":      int(trend_4h),
            "msb_detected":  int(msb_det),
        }

        # -- Gate 1: only LONG signals in 4H bull trend --
        if signal != 1 or trend_4h != 1:
            base_rec.update({"decision": "SKIP", "reason": "No long MSB or 4H not bull"})
            return base_rec

        # -- Gate 2: risk limits --
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            base_rec.update({"decision": "HALT", "reason": reason})
            alert(f"[RISK HALT] {reason}")
            return base_rec

        # -- Gate 3: max open positions --
        if count_open_positions(self.mt5) >= MAX_OPEN_POSITIONS:
            base_rec.update({"decision": "SKIP", "reason": "Max open positions reached"})
            return base_rec

        # -- ML Feature extraction --
        try:
            feat = extract_live_features(df, sig_i, self.meta)
        except Exception as e:
            log.warning(f"Feature extraction failed: {e}")
            base_rec.update({"decision": "SKIP", "reason": f"Feature error: {e}"})
            return base_rec

        # -- Inference --
        exp_prob  = float(self.model_a.predict_proba(feat)[0, 1])
        fake_prob = float(self.model_c.predict_proba(feat)[0, 1])
        cont_prob = float(self.model_d.predict_proba(feat)[0, 1])

        raw_score = W_EXP * exp_prob + W_CONT * cont_prob - W_FAKE * fake_prob

        # Normalized score: we need a population to rank against.
        # For live use, we use a simple min-max scale against the raw weights bounds.
        # Min possible: W_EXP*0 + W_CONT*0 - W_FAKE*1 = -0.40
        # Max possible: W_EXP*1 + W_CONT*1 - W_FAKE*0 = +0.60
        score_norm = (raw_score - (-W_FAKE)) / (W_EXP + W_CONT + W_FAKE)
        score_norm = max(0.0, min(1.0, score_norm))

        base_rec.update({
            "expansion_prob":     round(exp_prob, 4),
            "fake_breakout_prob": round(fake_prob, 4),
            "trend_prob":         round(cont_prob, 4),
            "ensemble_score":     round(raw_score, 4),
            "ensemble_score_norm": round(score_norm, 4),
        })

        # -- Gate 4: ensemble threshold --
        if score_norm < THRESHOLD:
            base_rec.update({
                "decision": "REJECT",
                "reason": f"Score {score_norm:.3f} < threshold {THRESHOLD}"
            })
            alert(f"[SIGNAL REJECTED] Score {score_norm:.3f} < {THRESHOLD} | bar: {bar_time}")
            return base_rec

        # -- Compute SL / TP --
        entry_idx = sig_i + 1
        if entry_idx >= len(df):
            base_rec.update({"decision": "SKIP", "reason": "No next bar for entry"})
            return base_rec

        entry_price = df["Open"].iloc[entry_idx]
        sl_price, tp_price = compute_sl_tp(
            direction=1,
            entry_price=entry_price,
            idx=sig_i,
            low_arr=df["Low"].values,
            high_arr=df["High"].values,
            atr_arr=df["atr14"].values,
            sl_lookback=CONFIG["sl_lookback"],
            sl_min_atr=CONFIG["sl_min_atr"],
            rr=RR,
        )

        # -- Lot sizing --
        acc = get_account_info(self.mt5)
        balance = acc.get("balance", self.risk.total_equity_start)

        lot, note = calc_lot_size(balance, entry_price, sl_price, self.min_lot, self.lot_step)
        if lot is None:
            base_rec.update({"decision": "SKIP", "reason": note})
            log.info(f"[LOT] {note}")
            return base_rec
        if note:
            log.info(f"[LOT] {note}")

        risk_pct = (abs(entry_price - sl_price) * lot * 100) / balance

        base_rec.update({
            "entry": round(entry_price, 2),
            "sl":    round(sl_price, 2),
            "tp":    round(tp_price, 2),
            "lot":   lot,
            "risk_pct": round(risk_pct, 3),
        })

        # -- Place order --
        result = place_order(self.mt5, entry_price, sl_price, tp_price, lot)
        if result is None:
            base_rec.update({"decision": "ERROR", "reason": "order_send failed"})
            return base_rec

        base_rec.update({"decision": "TRADE_OPENED", "reason": "All conditions met"})

        alert(
            f"[TRADE OPENED]\n"
            f"Entry : {entry_price:.2f}\n"
            f"SL    : {sl_price:.2f}\n"
            f"TP    : {tp_price:.2f}\n"
            f"Lot   : {lot}\n"
            f"Risk  : {risk_pct:.2f}%\n"
            f"Score : {score_norm:.3f}\n"
            f"Ticket: {result.get('order')}"
        )

        log_trade({
            "open_time": str(bar_time),
            "symbol":    SYMBOL,
            "direction": "LONG",
            "entry":     entry_price,
            "sl":        sl_price,
            "tp":        tp_price,
            "lot":       lot,
            "ticket":    result.get("order"),
        })

        return base_rec

    def run(self):
        self.startup()
        alert("[BOT] Entering polling loop. Checking every 60 seconds.")

        while True:
            try:
                # -- Fetch live bars --
                df_m15 = fetch_bars(self.mt5, SYMBOL, TF_M15, M15_BARS_NEEDED, "M15")
                df_4h  = fetch_bars(self.mt5, SYMBOL, TF_H4,  H4_BARS_NEEDED,  "4H")

                # -- Build pipeline --
                df = build_live_df(df_m15, df_4h)

                # -- Identify last CLOSED candle (exclude forming candle = last row) --
                closed_candle_idx  = len(df) - 2    # second to last
                closed_candle_time = df["Date"].iloc[closed_candle_idx]

                # -- Only evaluate if this is a NEW closed candle --
                if closed_candle_time == self.last_m15_time:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                self.last_m15_time = closed_candle_time
                log.info(f"[CANDLE] New M15 closed: {closed_candle_time}")
                alert(f"[CANDLE] {closed_candle_time} processed")

                # -- Evaluate signal --
                decision = self.evaluate_candle(df, closed_candle_idx)
                log_decision(decision)
                log.info(f"[DECISION] {decision.get('decision')} - {decision.get('reason')}")

                # -- Persist candle time immediately so restarts are safe --
                save_state({"last_processed_m15_candle_time": str(closed_candle_time)})
                log.info(f"[STATE] Saved last processed candle: {closed_candle_time}")

            except Exception as e:
                err_msg = traceback.format_exc()
                alert(f"[BOT ERROR]\n{err_msg[:1000]}")
                log.error(f"Loop error: {e}")

            time.sleep(POLL_INTERVAL_SEC)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if EXECUTION_MODE == "LIVE_DISABLED":
        print("[BLOCKED] EXECUTION_MODE=LIVE_DISABLED. Set PAPER_MT5_DEMO to run.")
        sys.exit(0)

    bot = LiveBot()
    bot.run()

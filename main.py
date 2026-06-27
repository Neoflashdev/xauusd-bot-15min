"""
main.py
=======
XAUUSD M15 MSB Live Paper Trading Bot.

FROZEN STRATEGY:
  - Long-only | 4H Bull Trend | Bullish MSB | Immediate Entry
  - Ensemble Score = 0.20 * Expansion + 0.40 * TrendCont - 0.40 * FakeBreakout
  - Threshold (normalized): >= THRESHOLD (default 0.65)
  - RR = 1.5
  - No training. Load frozen .pkl model files only.

PORTABLE DESIGN:
  - All settings loaded from config.py / .env
  - All paths relative to BASE (pathlib) — works on any server
  - MT5 credentials via .env — no code changes between Oracle / Google / Contabo

EXECUTION MODES:
  PAPER_INTERNAL   - print decisions only, no MT5 orders
  PAPER_MT5_DEMO   - send live demo orders to MT5 (paper money)
  LIVE_DISABLED    - blocked (requires explicit override)

MONITORING:
  - Telegram alert on every trade OPEN
  - Telegram alert on every trade CLOSE (with R-value, P&L, running stats)
  - Daily performance summary at midnight UTC
"""

import os
import sys
import time
import json
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
# CONFIG  (all settings from .env via config.py)
# -----------------------------------------------------------------------------
from config import (
    BASE, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    W_EXP, W_CONT, W_FAKE, THRESHOLD, RR, DIRECTION, SYMBOL,
    RISK_PCT, MAX_LOT, MAX_OPEN_POSITIONS,
    DAILY_LOSS_LIMIT_PCT, MAX_TOTAL_DRAWDOWN_PCT,
    STOP_AFTER_CONSEC_LOSSES, MAX_TRADES_PER_DAY,
    EXECUTION_MODE, POLL_INTERVAL_SEC,
    M15_BARS_NEEDED, H4_BARS_NEEDED,
    LOGS_DIR, DATA_DIR, MODELS_DIR,
    BOT_LOG_FILE, DECISION_LOG, TRADE_LOG, STATE_FILE, MODEL_PATHS,
    BACKTEST_WIN_RATE_LOW, BACKTEST_WIN_RATE_HIGH,
    BACKTEST_PF_LOW, BACKTEST_PF_HIGH,
    BACKTEST_AVG_R, BACKTEST_MAX_DD_R,
    BACKTEST_TRADES_MONTH, MIN_TRADES_FOR_WARN,
)

# MT5 timeframes (set after import)
TF_M15 = None
TF_H4  = None

# Magic number identifies our bot's trades in MT5
MAGIC = 20240001

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
_fh = logging.FileHandler(str(BOT_LOG_FILE), encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])
log = logging.getLogger("LiveBot")

# -----------------------------------------------------------------------------
# STATE FILE HELPERS  (restart-safe candle tracking + processed deal IDs)
# -----------------------------------------------------------------------------
def load_state() -> dict:
    """Load state.json. Returns empty dict if file does not exist."""
    if STATE_FILE.exists():
        try:
            with open(str(STATE_FILE), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"[STATE] Could not read state.json: {e}")
    return {}


def save_state(state: dict):
    """Atomically write state dict to state.json."""
    try:
        tmp = str(STATE_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, str(STATE_FILE))
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
        requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10
        )
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
    "open_time", "close_time", "symbol", "direction",
    "entry", "sl", "tp", "lot",
    "result", "pnl_usd", "r_value",
    "ticket", "position_id",
]


def _ensure_csv(path: Path, cols: list):
    if not path.exists():
        with open(str(path), "w", newline="") as f:
            csv.writer(f).writerow(cols)


def log_decision(row: dict):
    _ensure_csv(DECISION_LOG, DECISION_COLS)
    with open(str(DECISION_LOG), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DECISION_COLS)
        w.writerow({k: row.get(k, "") for k in DECISION_COLS})


def log_trade(row: dict):
    _ensure_csv(TRADE_LOG, TRADE_COLS)
    with open(str(TRADE_LOG), "a", newline="") as f:
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

    # Pass credentials if provided (required for cloud servers with fresh MT5 install)
    init_kwargs = {}
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        init_kwargs = {
            "login":    MT5_LOGIN,
            "password": MT5_PASSWORD,
            "server":   MT5_SERVER,
        }

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        alert(f"[BOT] MT5 init failed: {err}")
        sys.exit(1)

    # Ensure XAUUSD is in Market Watch
    if not mt5.symbol_select(SYMBOL, True):
        alert(f"[BOT] Could not select {SYMBOL} in Market Watch")
        mt5.shutdown()
        sys.exit(1)

    alert("[BOT] MT5 connected [OK]")
    return mt5


# -----------------------------------------------------------------------------
# LIVE DATA LOADING
# -----------------------------------------------------------------------------
def fetch_bars(mt5, symbol: str, timeframe, n_bars: int, label: str) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch {label} bars: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["Date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"
    }, inplace=True)
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
        "login":    acc.login,
        "server":   acc.server,
        "balance":  acc.balance,
        "equity":   acc.equity,
        "currency": acc.currency,
    }


# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
def load_models() -> tuple:
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            alert(
                f"[BOT] Model file missing: {path}\n"
                f"Run train_models.py first."
            )
            sys.exit(1)

    with open(str(MODEL_PATHS["expansion"]), "rb") as f:
        model_a = pickle.load(f)
    with open(str(MODEL_PATHS["fake"]), "rb") as f:
        model_c = pickle.load(f)
    with open(str(MODEL_PATHS["trend"]), "rb") as f:
        model_d = pickle.load(f)
    with open(str(MODEL_PATHS["metadata"]), "rb") as f:
        meta = pickle.load(f)

    log.info(f"  Models loaded. Features: {meta['n_features']}")
    log.info(f"  Trained on: {meta['date_range']['start']} -> {meta['date_range']['end']}")
    return model_a, model_c, model_d, meta


# -----------------------------------------------------------------------------
# INDICATOR PIPELINE  (reused identically from backtest — DO NOT MODIFY)
# -----------------------------------------------------------------------------
sys.path.insert(0, str(BASE))
from BTCUSD_Backtest import (
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

    allow_l = DIRECTION in ["LONG", "BOTH"]
    allow_s = DIRECTION in ["SHORT", "BOTH"]
    df = generate_signals(df, allow_longs=allow_l, allow_shorts=allow_s)
    return df


# -----------------------------------------------------------------------------
# LIVE FEATURE EXTRACTION  (single candle — identical to backtest)
# DO NOT MODIFY: any change here breaks model parity with backtested results.
# -----------------------------------------------------------------------------
def extract_live_features(df: pd.DataFrame, sig_i: int, meta: dict) -> np.ndarray:
    """
    Extract the exact same V2 feature vector for a single bar (sig_i).
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

    row["LIQ_Dist_5D_H"]  = (C.iloc[sig_i] - H.rolling(5*96).max().shift(1).iloc[sig_i])  / ATR.iloc[sig_i]
    row["LIQ_Dist_5D_L"]  = (C.iloc[sig_i] - L.rolling(5*96).min().shift(1).iloc[sig_i])  / ATR.iloc[sig_i]
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
    row["VOL_Is_NR4"]       = int(tr.iloc[sig_i] < tr.rolling(4).min().shift(1).iloc[sig_i])
    row["VOL_Is_NR7"]       = int(tr.iloc[sig_i] < tr.rolling(7).min().shift(1).iloc[sig_i])
    row["VOL_Inside_Bar"]   = int(H.iloc[sig_i] < H.iloc[sig_i-1] and L.iloc[sig_i] > L.iloc[sig_i-1])
    inside_bar_s            = ((H < H.shift(1)) & (L > L.shift(1))).astype(int)
    row["VOL_Inside_Streak"] = inside_bar_s.groupby((inside_bar_s == 0).cumsum()).cumcount().iloc[sig_i]

    # -- GROUP 4: GEOMETRY (GEO_) --
    hl_range = (H - L).replace(0, np.nan)
    row["GEO_Body_Pct"]     = ((C - O).abs() / hl_range).iloc[sig_i]
    row["GEO_Up_Wick_Pct"]  = ((H - df[["Open","Close"]].max(axis=1)) / hl_range).iloc[sig_i]
    row["GEO_Dn_Wick_Pct"]  = ((df[["Open","Close"]].min(axis=1) - L) / hl_range).iloc[sig_i]
    row["GEO_Retracement_3B"]   = (df["High"].rolling(3).max() - df["Low"].rolling(3).min()).iloc[sig_i] / ATR.iloc[sig_i]
    row["GEO_Consecutive_Bull"] = (C > O).astype(int).groupby((C <= O).cumsum()).cumcount().iloc[sig_i]
    row["GEO_Consecutive_Bear"] = (C < O).astype(int).groupby((C >= O).cumsum()).cumcount().iloc[sig_i]
    row["GEO_Momentum_Score"]   = ((C - C.shift(5)) / ATR).iloc[sig_i]

    sw_h_val = last_sw_h.iloc[sig_i]
    sw_l_val = last_sw_l.iloc[sig_i]
    c_price  = C.iloc[sig_i]
    atr_val  = ATR.iloc[sig_i] if not pd.isna(ATR.iloc[sig_i]) else 1.0

    row["GEO_MSB_Dist_Swing_H"]  = (sw_h_val - c_price) / atr_val if not pd.isna(sw_h_val) else 0
    row["GEO_MSB_Dist_Swing_L"]  = (c_price - sw_l_val) / atr_val if not pd.isna(sw_l_val) else 0
    sw_size_val = (sw_h_val - sw_l_val) / atr_val if (not pd.isna(sw_h_val) and not pd.isna(sw_l_val)) else 1
    row["GEO_Breakout_Size"]      = (c_price - sw_h_val) / atr_val if not pd.isna(sw_h_val) else 0
    row["GEO_Breakout_vs_Swing"]  = row["GEO_Breakout_Size"] / sw_size_val if sw_size_val > 0 else 0

    # Build feature vector in exact same column order as training
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
def calc_lot_size(
    balance: float, entry: float, sl: float,
    min_lot: float, lot_step: float,
    contract_size: float = 100.0
) -> tuple:
    """
    Calculate lot size so risk at SL = RISK_PCT * balance.
    XAUUSD: 1 lot = 100 oz
    """
    risk_amount = balance * RISK_PCT
    stop_dist   = abs(entry - sl)
    if stop_dist <= 0:
        return None, "SL distance is zero"

    raw_lot = risk_amount / (stop_dist * contract_size)
    lot     = max(min_lot, round(raw_lot // lot_step * lot_step, 8))

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
        return {"retcode": 0, "order": 99999, "position_id": 99999}

    if EXECUTION_MODE == "LIVE_DISABLED":
        alert("[BOT] Live trading is disabled. EXECUTION_MODE=LIVE_DISABLED")
        return None

    # PAPER_MT5_DEMO: real demo order
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        log.error("Could not get tick data.")
        return None

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       float(lot),
        "type":         mt5.ORDER_TYPE_BUY,
        "price":        tick.ask,
        "sl":           round(sl, 2),
        "tp":           round(tp, 2),
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      "XAUUSD_MSB_Bot",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error(f"order_send returned None: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed. retcode={result.retcode} | comment={result.comment}")
        return None

    # result.deal is the opening deal ticket = position_id for the first deal
    return {
        "retcode":     result.retcode,
        "order":       result.order,
        "position_id": result.deal,    # MT5: first deal ticket == position_id
    }


# -----------------------------------------------------------------------------
# OPEN POSITION CHECK
# -----------------------------------------------------------------------------
def count_open_positions(mt5) -> int:
    if EXECUTION_MODE == "PAPER_INTERNAL":
        return 0
    positions = mt5.positions_get(symbol=SYMBOL)
    return len(positions) if positions else 0


# -----------------------------------------------------------------------------
# RISK STATE
# -----------------------------------------------------------------------------
class RiskState:
    def __init__(self):
        self.reset_day()
        self.total_equity_start = None
        self.total_drawdown_pct = 0.0

    def reset_day(self):
        """Called at Midnight UTC daily reset."""
        self.daily_pnl_pct   = 0.0
        self.trades_today    = 0
        self.consec_losses   = 0
        self.halted_today    = False
        self.last_reset_date = datetime.datetime.utcnow().date()

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
        self.daily_pnl_pct     += pnl_pct
        self.total_drawdown_pct = min(self.total_drawdown_pct, self.daily_pnl_pct)
        self.trades_today       += 1
        if pnl_pct < 0:
            self.consec_losses += 1
        else:
            self.consec_losses  = 0


# -----------------------------------------------------------------------------
# LIVE BOT
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
        self._last_summary_date = None

        # Load persisted candle time (restart-safe)
        state = load_state()
        saved = state.get("last_processed_m15_candle_time")
        if saved:
            try:
                self.last_m15_time = pd.Timestamp(saved)
                log.info(f"[STATE] Resumed. Last processed candle: {self.last_m15_time}")
            except Exception:
                log.warning("[STATE] Could not parse saved candle time. Starting fresh.")

    # -------------------------------------------------------------------------
    # STARTUP
    # -------------------------------------------------------------------------
    def startup(self):
        log.info("=" * 60)
        log.info(f"  {SYMBOL} M15 MSB LIVE BOT")
        log.info(f"  Mode     : {EXECUTION_MODE} | Live: {os.getenv('LIVE_ENABLED')}")
        log.info(f"  Symbol   : {SYMBOL}")
        log.info(f"  Strategy : Direction: {DIRECTION} | MSB | Score >= {THRESHOLD} | RR={RR}")
        log.info(f"  Risk     : {RISK_PCT*100:.2f}% per trade | MaxLot: {MAX_LOT}")
        log.info("=" * 60)

        self.mt5 = init_mt5()

        acc = get_account_info(self.mt5)
        self.risk.total_equity_start = acc.get("balance", 10000)
        self.min_lot, self.lot_step  = get_broker_lot_info(self.mt5)
        
        # Validate broker symbol info
        sym_info = self.mt5.symbol_info(SYMBOL)
        if sym_info is None:
            log.error(f"[ERROR] Symbol {SYMBOL} not found in MT5!")
            sys.exit(1)
            
        log.info("\n--- MT5 SYMBOL INFO ---")
        log.info(f"symbol        : {sym_info.name}")
        log.info(f"digits        : {sym_info.digits}")
        log.info(f"point         : {sym_info.point}")
        log.info(f"contract_size : {sym_info.trade_contract_size}")
        log.info(f"min_lot       : {sym_info.volume_min}")
        log.info(f"max_lot       : {sym_info.volume_max}")
        log.info(f"lot_step      : {sym_info.volume_step}")
        log.info(f"trade_mode    : {sym_info.trade_mode}")
        log.info(f"spread        : {sym_info.spread}")
        log.info("-----------------------\n")

        alert(
            f"[BOT STARTED]\n"
            f"Account : {acc.get('login')}\n"
            f"Server  : {acc.get('server')}\n"
            f"Balance : {acc.get('balance')} {acc.get('currency')}\n"
            f"Symbol  : {SYMBOL}\n"
            f"Mode    : {EXECUTION_MODE}\n"
            f"Risk    : {RISK_PCT*100:.2f}% | MaxLot: {MAX_LOT}"
        )

        self.model_a, self.model_c, self.model_d, self.meta = load_models()
        alert(
            f"[BOT] Models loaded [OK]\n"
            f"Features: {self.meta['n_features']}\n"
            f"Trained : {self.meta['date_range']['start']} -> {self.meta['date_range']['end']}"
        )

        log.info(f"  Open positions : {count_open_positions(self.mt5)}")

    # -------------------------------------------------------------------------
    # CANDLE EVALUATION
    # -------------------------------------------------------------------------
    def evaluate_candle(self, df: pd.DataFrame, sig_i: int) -> dict:
        """Evaluate one closed M15 candle. Returns decision dict."""
        bar_time = df["Date"].iloc[sig_i]
        signal   = df["signal"].iloc[sig_i]
        trend_4h = df["4h_trend"].iloc[sig_i]
        msb_det  = bool(df["bull_msb"].iloc[sig_i])

        base_rec = {
            "time":             datetime.datetime.utcnow().isoformat(),
            "symbol":           SYMBOL,
            "m15_candle_time":  str(bar_time),
            "signal":           int(signal),
            "trend_4h":         int(trend_4h),
            "msb_detected":     int(msb_det),
        }

        # Gate 1: Valid signal detection (LONG=1, SHORT=-1)
        if signal == 0:
            base_rec.update({"decision": "SKIP", "reason": "No valid signal"})
            return base_rec

        # Gate 2: Risk limits
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            base_rec.update({"decision": "HALT", "reason": reason})
            alert(f"[RISK HALT] {reason}")
            return base_rec

        # Gate 3: Max open positions
        if count_open_positions(self.mt5) >= MAX_OPEN_POSITIONS:
            base_rec.update({"decision": "SKIP", "reason": "Max open positions reached"})
            return base_rec

        # ML Feature extraction
        try:
            feat = extract_live_features(df, sig_i, self.meta)
        except Exception as e:
            log.warning(f"Feature extraction failed: {e}")
            base_rec.update({"decision": "SKIP", "reason": f"Feature error: {e}"})
            return base_rec

        # Inference
        exp_prob  = float(self.model_a.predict_proba(feat)[0, 1])
        fake_prob = float(self.model_c.predict_proba(feat)[0, 1])
        cont_prob = float(self.model_d.predict_proba(feat)[0, 1])

        raw_score  = W_EXP * exp_prob + W_CONT * cont_prob - W_FAKE * fake_prob
        score_norm = (raw_score - (-W_FAKE)) / (W_EXP + W_CONT + W_FAKE)
        score_norm = max(0.0, min(1.0, score_norm))

        base_rec.update({
            "expansion_prob":      round(exp_prob, 4),
            "fake_breakout_prob":  round(fake_prob, 4),
            "trend_prob":          round(cont_prob, 4),
            "ensemble_score":      round(raw_score, 4),
            "ensemble_score_norm": round(score_norm, 4),
        })

        # Gate 4: Ensemble threshold
        if score_norm < THRESHOLD:
            base_rec.update({
                "decision": "REJECT",
                "reason":   f"Score {score_norm:.3f} < threshold {THRESHOLD}"
            })
            alert(f"[SIGNAL REJECTED] Score {score_norm:.3f} < {THRESHOLD} | bar: {bar_time}")
            return base_rec

        # Compute SL / TP
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

        # Lot sizing
        acc     = get_account_info(self.mt5)
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
            "entry":    round(entry_price, 2),
            "sl":       round(sl_price, 2),
            "tp":       round(tp_price, 2),
            "lot":      lot,
            "risk_pct": round(risk_pct, 3),
        })

        # Place order
        result = place_order(self.mt5, entry_price, sl_price, tp_price, lot)
        if result is None:
            base_rec.update({"decision": "ERROR", "reason": "order_send failed"})
            return base_rec

        position_id = result.get("position_id")
        base_rec.update({"decision": "TRADE_OPENED", "reason": "All conditions met"})

        alert(
            f"[TRADE OPENED]\n"
            f"Bar    : {bar_time}\n"
            f"Entry  : {entry_price:.2f}\n"
            f"SL     : {sl_price:.2f}\n"
            f"TP     : {tp_price:.2f}\n"
            f"Lot    : {lot}\n"
            f"Risk   : {risk_pct:.2f}%\n"
            f"Score  : {score_norm:.3f}\n"
            f"Ticket : {result.get('order')}"
        )

        log_trade({
            "open_time":   str(bar_time),
            "symbol":      SYMBOL,
            "direction":   "LONG",
            "entry":       entry_price,
            "sl":          sl_price,
            "tp":          tp_price,
            "lot":         lot,
            "ticket":      result.get("order"),
            "position_id": position_id,
        })

        return base_rec

    # -------------------------------------------------------------------------
    # TRADE CLOSE MONITORING
    # -------------------------------------------------------------------------
    def check_closed_positions(self):
        """
        Poll MT5 deal history for closed positions not yet recorded.
        For each newly closed trade: update trade_log.csv, update RiskState,
        send detailed Telegram alert with running performance stats.
        """
        if EXECUTION_MODE == "PAPER_INTERNAL":
            return

        now      = datetime.datetime.utcnow()
        from_dt  = now - datetime.timedelta(hours=48)

        try:
            deals = self.mt5.history_deals_get(from_dt, now)
        except Exception as e:
            log.warning(f"[CLOSE_CHECK] history_deals_get failed: {e}")
            return

        if deals is None or len(deals) == 0:
            return

        # Filter: our magic number + OUT entry (closing a trade)
        try:
            close_deals = [
                d for d in deals
                if d.magic == MAGIC
                and d.entry == self.mt5.DEAL_ENTRY_OUT
                and d.symbol == SYMBOL
            ]
        except Exception:
            return

        if not close_deals:
            return

        state = load_state()
        processed_deal_ids = set(state.get("processed_deal_ids", []))

        changed = False
        for deal in close_deals:
            if deal.ticket in processed_deal_ids:
                continue

            changed = True
            processed_deal_ids.add(deal.ticket)

            # Find matching open trade by position_id
            trade_info = self._find_open_trade(deal.position_id)

            # Calculate R-value
            r_value = 0.0
            if trade_info:
                try:
                    entry     = float(trade_info.get("entry", 0))
                    sl        = float(trade_info.get("sl", 0))
                    stop_dist = abs(entry - sl)
                    if stop_dist > 0:
                        move    = deal.price - entry
                        r_value = move / stop_dist
                except Exception:
                    r_value = 0.0

            pnl        = deal.profit
            result_str = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
            r_sign     = "+" if r_value >= 0 else ""

            close_time = str(datetime.datetime.fromtimestamp(deal.time))

            # Update trade_log.csv
            self._update_trade_log(deal.position_id, {
                "close_time": close_time,
                "result":     result_str,
                "pnl_usd":    round(pnl, 2),
                "r_value":    round(r_value, 2),
            })

            # Update RiskState with actual P&L
            if self.risk.total_equity_start:
                pnl_pct = (pnl / self.risk.total_equity_start) * 100
                self.risk.record_trade(pnl_pct)

            # Calculate running performance stats
            stats = self._calc_running_stats()

            # Send Telegram close alert
            self._send_trade_close_alert(deal, trade_info, r_value, result_str, stats)

            log.info(
                f"[TRADE CLOSED] {result_str} | "
                f"Ticket={deal.ticket} | R={r_sign}{r_value:.2f} | P&L=${pnl:+.2f}"
            )

        if changed:
            state["processed_deal_ids"] = list(processed_deal_ids)
            save_state(state)

    def _find_open_trade(self, position_id) -> dict:
        """Find an open trade in trade_log.csv by position_id."""
        if not TRADE_LOG.exists():
            return {}
        try:
            df = pd.read_csv(str(TRADE_LOG))
            if "position_id" not in df.columns:
                return {}
            row = df[df["position_id"] == position_id]
            if len(row) == 0:
                return {}
            return row.iloc[0].to_dict()
        except Exception:
            return {}

    def _update_trade_log(self, position_id, updates: dict):
        """Update a trade row in trade_log.csv with close info."""
        if not TRADE_LOG.exists():
            return
        try:
            df = pd.read_csv(str(TRADE_LOG))
            if "position_id" not in df.columns:
                return
            mask = df["position_id"] == position_id
            for col, val in updates.items():
                if col in df.columns:
                    df.loc[mask, col] = val
            df.to_csv(str(TRADE_LOG), index=False)
        except Exception as e:
            log.warning(f"[TRADE_LOG] Could not update close info: {e}")

    def _calc_running_stats(self) -> dict:
        """Calculate running performance metrics from trade_log.csv."""
        try:
            df = pd.read_csv(str(TRADE_LOG))
            closed = df[df["result"].notna() & (df["result"] != "")]
            n = len(closed)
            if n == 0:
                return {"trades": 0}

            wins   = closed[closed["result"] == "WIN"]
            losses = closed[closed["result"] == "LOSS"]

            win_rate     = len(wins) / n * 100
            gross_profit = wins["pnl_usd"].sum() if len(wins) > 0 else 0.0
            gross_loss   = abs(losses["pnl_usd"].sum()) if len(losses) > 0 else 0.0
            pf           = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

            if "r_value" in closed.columns:
                avg_r = float(closed["r_value"].mean())
                # Max drawdown in R
                r_vals = closed["r_value"].values.astype(float)
                cum_r  = np.cumsum(r_vals)
                run_max = np.maximum.accumulate(cum_r)
                dd      = cum_r - run_max
                max_dd_r = float(dd.min()) if len(dd) > 0 else 0.0
            else:
                avg_r    = 0.0
                max_dd_r = 0.0

            return {
                "trades":   n,
                "win_rate": round(win_rate, 1),
                "pf":       pf,
                "avg_r":    round(avg_r, 3),
                "max_dd_r": round(max_dd_r, 2),
            }
        except Exception:
            return {"trades": 0}

    def _send_trade_close_alert(self, deal, trade_info, r_value, result_str, stats):
        """Send Telegram message with trade close details + running stats vs targets."""
        entry    = trade_info.get("entry", "?") if trade_info else "?"
        sl       = trade_info.get("sl",    "?") if trade_info else "?"
        tp       = trade_info.get("tp",    "?") if trade_info else "?"
        r_sign   = "+" if r_value >= 0 else ""

        n        = stats.get("trades", 0)
        wr       = stats.get("win_rate", 0.0)
        pf       = stats.get("pf", 0.0)
        avg_r    = stats.get("avg_r", 0.0)
        max_dd_r = stats.get("max_dd_r", 0.0)

        # Status tag vs backtest targets (only meaningful after MIN_TRADES)
        def tag(val, lo, hi):
            if n < MIN_TRADES_FOR_WARN:
                return "(building)"
            return "[OK]" if lo <= val <= hi else "[WATCH]"

        wr_tag  = tag(wr,  BACKTEST_WIN_RATE_LOW, BACKTEST_WIN_RATE_HIGH)
        pf_tag  = tag(pf,  BACKTEST_PF_LOW, BACKTEST_PF_HIGH)
        r_tag   = tag(avg_r, BACKTEST_AVG_R * 0.5, BACKTEST_AVG_R * 1.5)
        dd_tag  = "[OK]" if max_dd_r >= BACKTEST_MAX_DD_R else "[WATCH]"

        msg = (
            f"[TRADE CLOSED] {result_str}\n"
            f"Ticket : {deal.ticket}\n"
            f"Entry  : {entry}\n"
            f"Exit   : {deal.price:.2f}\n"
            f"SL/TP  : {sl} / {tp}\n"
            f"R      : {r_sign}{r_value:.2f}R\n"
            f"P&L    : ${deal.profit:+.2f}\n"
            f"\n--- Running Stats ({n} trades) ---\n"
            f"Win%   : {wr:.1f}%   Target 46-50%  {wr_tag}\n"
            f"PF     : {pf:.2f}    Target 1.50-1.73 {pf_tag}\n"
            f"Avg R  : {r_sign}{avg_r:.3f}R  Target +0.39R {r_tag}\n"
            f"Max DD : {max_dd_r:.1f}R   Limit -6.0R  {dd_tag}"
        )
        alert(msg)

    # -------------------------------------------------------------------------
    # TELEGRAM COMMAND LISTENER
    # -------------------------------------------------------------------------
    def check_telegram_commands(self):
        """Poll Telegram getUpdates API for commands like /report or /status."""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
            
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        state = load_state()
        last_id = state.get("last_telegram_update_id")
        
        params = {"timeout": 5}
        if last_id:
            params["offset"] = last_id + 1
            
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            if data.get("ok"):
                for update in data.get("result", []):
                    uid = update["update_id"]
                    state["last_telegram_update_id"] = uid
                    save_state(state)
                    
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip().lower()
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    
                    if not text:
                        continue
                        
                    # Log that we saw a message (helpful for debugging)
                    log.info(f"[TELEGRAM DEBUG] Saw message '{text}' from chat_id {chat_id}")
                    
                    # Check if the text starts with any of our commands
                    is_command = text.startswith("/status") or text.startswith("/report") or text.startswith("/stats")
                    
                    if is_command:
                        if chat_id == str(TELEGRAM_CHAT_ID).strip():
                            log.info(f"[TELEGRAM] Authorized command received: {text}")
                            self._send_daily_summary()
                        else:
                            log.warning(f"[TELEGRAM] Unauthorized command from chat_id {chat_id} (Expected {TELEGRAM_CHAT_ID})")
        except Exception as e:
            # Uncomment for debugging if needed: log.error(f"Telegram polling error: {e}")
            pass

    # -------------------------------------------------------------------------
    # DAILY SUMMARY
    # -------------------------------------------------------------------------
    def _send_daily_summary(self):
        """Send daily performance vs backtest target report via Telegram."""
        stats = self._calc_running_stats()
        n     = stats.get("trades", 0)
        wr    = stats.get("win_rate", 0.0)
        pf    = stats.get("pf", 0.0)
        avg_r = stats.get("avg_r", 0.0)
        dd    = stats.get("max_dd_r", 0.0)

        def row(label, live_val, target_str):
            return f"{label:<14}: {live_val:<10} Target: {target_str}"

        msg = (
            f"[DAILY REPORT] {datetime.datetime.utcnow().date()}\n"
            f"{'='*38}\n"
            f"{row('Win Rate',   f'{wr:.1f}%',    '46-50%')}\n"
            f"{row('Prof Factor', f'{pf:.2f}',    '1.50-1.73')}\n"
            f"{row('Avg R/trade', f'{avg_r:+.3f}R', '+0.39R')}\n"
            f"{row('Max DD',      f'{dd:.1f}R',   '-6.0R max')}\n"
            f"{row('Trades',      str(n),          f'~{BACKTEST_TRADES_MONTH}/month')}\n"
            f"{'='*38}"
        )
        alert(msg)

    # -------------------------------------------------------------------------
    # MAIN POLLING LOOP
    # -------------------------------------------------------------------------
    def run(self):
        self.startup()
        alert("[BOT] Entering polling loop. Checking every 60 seconds.")

        while True:
            try:
                # -- Check for closed trades every loop --
                self.check_closed_positions()

                # -- Check for incoming Telegram commands --
                self.check_telegram_commands()

                # -- Daily summary at midnight UTC --
                today = datetime.datetime.utcnow().date()
                if self._last_summary_date is None:
                    self._last_summary_date = today
                elif today != self._last_summary_date:
                    self._last_summary_date = today
                    self._send_daily_summary()

                # -- Fetch live bars --
                df_m15 = fetch_bars(self.mt5, SYMBOL, TF_M15, M15_BARS_NEEDED, "M15")
                df_4h  = fetch_bars(self.mt5, SYMBOL, TF_H4,  H4_BARS_NEEDED,  "4H")

                # -- Build pipeline --
                df = build_live_df(df_m15, df_4h)

                # -- Identify last CLOSED candle (exclude forming = last row) --
                closed_candle_idx  = len(df) - 2
                closed_candle_time = df["Date"].iloc[closed_candle_idx]

                # -- Only evaluate if this is a NEW closed candle --
                if closed_candle_time == self.last_m15_time:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                self.last_m15_time = closed_candle_time
                log.info(f"[CANDLE] New M15 closed: {closed_candle_time}")

                # -- Evaluate signal --
                decision = self.evaluate_candle(df, closed_candle_idx)
                log_decision(decision)
                log.info(f"[DECISION] {decision.get('decision')} - {decision.get('reason')}")

                # -- Persist candle time (restart-safe) --
                state = load_state()
                state["last_processed_m15_candle_time"] = str(closed_candle_time)
                save_state(state)
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
        print("[BLOCKED] EXECUTION_MODE=LIVE_DISABLED. Set PAPER_MT5_DEMO in .env to run.")
        sys.exit(0)

    bot = LiveBot()
    bot.run()

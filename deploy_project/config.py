"""
config.py
=========
Central configuration for the XAUUSD M15 MSB Live Bot.

All settings are loaded from the .env file in this directory.
On any new server: clone repo, copy .env, run main.py.
No Python code changes needed between servers.

DO NOT hardcode values in any other file.
Import from here: `from config import *`
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------
# Auto-load .env if python-dotenv is installed
# -----------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    env_filename = os.getenv("ENV_FILE", ".env")
    _env_path = Path(__file__).parent / env_filename
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed; os.environ still works

# -----------------------------------------------------------------------
# BASE DIRECTORY  (works on Windows, Linux, any path)
# -----------------------------------------------------------------------
BASE = Path(__file__).parent

# -----------------------------------------------------------------------
# MT5 CREDENTIALS  (for cloud auto-login)
# -----------------------------------------------------------------------
MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "")

# -----------------------------------------------------------------------
# TELEGRAM
# -----------------------------------------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# -----------------------------------------------------------------------
# FROZEN STRATEGY CONSTANTS  (DO NOT CHANGE without backtest validation)
# -----------------------------------------------------------------------
W_EXP     = float(os.getenv("W_EXP", "0.20"))   # Ensemble weight: Expansion model
W_CONT    = float(os.getenv("W_CONT", "0.40"))  # Ensemble weight: Trend Continuation model
W_FAKE    = float(os.getenv("W_FAKE", "0.40"))  # Ensemble weight: Fake Breakout model (subtracted)
THRESHOLD = float(os.getenv("THRESHOLD", "0.65"))   # Normalized ensemble score minimum
RR        = float(os.getenv("RR", "1.5"))           # Risk:Reward ratio
DIRECTION = os.getenv("DIRECTION", "LONG")          # Trading direction (LONG, SHORT, or BOTH)
SYMBOL           = os.getenv("SYMBOL", "XAUUSD")
BACKTEST_MODULE  = os.getenv("BACKTEST_MODULE", "XAUUSD_Colab_Backtest")

# -----------------------------------------------------------------------
# RISK SETTINGS
# -----------------------------------------------------------------------
# RISK_PCT is expressed as a percentage in .env (e.g. "0.5" = 0.5% per trade)
RISK_PCT                  = float(os.getenv("RISK_PCT", "0.5")) / 100.0
MAX_LOT                   = float(os.getenv("MAX_LOT", "0.20"))
MAX_OPEN_POSITIONS        = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
DAILY_LOSS_LIMIT_PCT      = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "2.0"))
MAX_TOTAL_DRAWDOWN_PCT    = float(os.getenv("MAX_TOTAL_DRAWDOWN_PCT", "6.0"))
STOP_AFTER_CONSEC_LOSSES  = int(os.getenv("STOP_AFTER_CONSEC_LOSSES", "2"))
MAX_TRADES_PER_DAY        = int(os.getenv("MAX_TRADES_PER_DAY", "2"))

# -----------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------
EXECUTION_MODE    = os.getenv("EXECUTION_MODE", "PAPER_MT5_DEMO")
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "60"))

# -----------------------------------------------------------------------
# DATA LOOKBACK
# -----------------------------------------------------------------------
# Default fallback values (can be overridden via .env)
M15_BARS_NEEDED      = int(os.getenv("M15_BARS_NEEDED", "4000"))
H4_BARS_NEEDED       = int(os.getenv("H4_BARS_NEEDED", "500"))

# -----------------------------------------------------------------------
# DIRECTORY STRUCTURE  (portable, relative to this file)
# -----------------------------------------------------------------------
LOGS_DIR   = BASE / "logs"
DATA_DIR   = BASE / "data"
MODELS_DIR = BASE / "models" / SYMBOL.lower()

# Auto-create directories so the bot runs on a fresh server
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# FILE PATHS
# -----------------------------------------------------------------------
BOT_LOG_FILE  = LOGS_DIR / f"{SYMBOL}_live_bot.log"
DECISION_LOG  = LOGS_DIR / f"{SYMBOL}_decision_log.csv"
TRADE_LOG     = LOGS_DIR / f"{SYMBOL}_trade_log.csv"
STATE_FILE    = DATA_DIR / f"state_{SYMBOL}.json"

MODEL_PATHS = {
    "expansion": MODELS_DIR / "expansion_model.pkl",
    "fake":      MODELS_DIR / "fake_breakout_model.pkl",
    "trend":     MODELS_DIR / "trend_continuation_model.pkl",
    "metadata":  MODELS_DIR / "feature_metadata.pkl",
}

# -----------------------------------------------------------------------
# BACKTEST REFERENCE TARGETS  (used by performance_tracker.py)
# -----------------------------------------------------------------------
BACKTEST_WIN_RATE_LOW  = float(os.getenv("BACKTEST_WIN_RATE_LOW", "46.0"))
BACKTEST_WIN_RATE_HIGH = float(os.getenv("BACKTEST_WIN_RATE_HIGH", "50.0"))
BACKTEST_PF_LOW        = float(os.getenv("BACKTEST_PF_LOW", "1.50"))
BACKTEST_PF_HIGH       = float(os.getenv("BACKTEST_PF_HIGH", "1.73"))
BACKTEST_AVG_R         = float(os.getenv("BACKTEST_AVG_R", "0.39"))
BACKTEST_MAX_DD_R      = float(os.getenv("BACKTEST_MAX_DD_R", "-6.0"))
BACKTEST_TRADES_MONTH  = int(os.getenv("BACKTEST_TRADES_MONTH", "14"))
MIN_TRADES_FOR_WARN    = int(os.getenv("MIN_TRADES_FOR_WARN", "20"))

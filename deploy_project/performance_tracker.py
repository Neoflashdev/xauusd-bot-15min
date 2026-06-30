"""
performance_tracker.py
======================
Standalone script. Run anytime to see how your live paper trading
compares against the frozen backtest targets.

Usage:
    python performance_tracker.py
    python performance_tracker.py --telegram     (also sends to Telegram)

Reads: data/live_trade_log.csv
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

# Load config for paths and targets
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TRADE_LOG,
    BACKTEST_WIN_RATE_LOW, BACKTEST_WIN_RATE_HIGH,
    BACKTEST_PF_LOW, BACKTEST_PF_HIGH,
    BACKTEST_AVG_R, BACKTEST_MAX_DD_R,
    BACKTEST_TRADES_MONTH, MIN_TRADES_FOR_WARN,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
)


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        print("  Sent to Telegram.")
    except Exception as e:
        print(f"  Telegram send failed: {e}")


def status(val, lo, hi, invert=False) -> str:
    """Return [OK] or [WATCH] based on whether val is in [lo, hi]."""
    ok = lo <= val <= hi
    if invert:
        ok = not ok
    return "[OK]   " if ok else "[WATCH]"


# -----------------------------------------------------------------------
# LOAD AND CALCULATE
# -----------------------------------------------------------------------
def load_closed_trades() -> pd.DataFrame:
    if not TRADE_LOG.exists():
        print(f"[ERROR] Trade log not found: {TRADE_LOG}")
        sys.exit(1)

    df = pd.read_csv(str(TRADE_LOG))

    # Filter to closed trades only (result column filled)
    if "result" not in df.columns:
        print("[ERROR] 'result' column missing from trade log. No closed trades yet.")
        sys.exit(0)

    closed = df[df["result"].notna() & (df["result"].astype(str).str.strip() != "")]
    return closed


def calc_stats(df: pd.DataFrame) -> dict:
    n      = len(df)
    wins   = df[df["result"] == "WIN"]
    losses = df[df["result"] == "LOSS"]

    win_rate     = len(wins) / n * 100 if n > 0 else 0.0
    gross_profit = wins["pnl_usd"].astype(float).sum()   if len(wins)   > 0 else 0.0
    gross_loss   = abs(losses["pnl_usd"].astype(float).sum()) if len(losses) > 0 else 0.0
    pf           = gross_profit / gross_loss if gross_loss > 0 else 0.0
    net_pnl      = float(df["pnl_usd"].astype(float).sum())

    if "r_value" in df.columns and n > 0:
        r_vals    = df["r_value"].astype(float).values
        avg_r     = float(r_vals.mean())
        total_r   = float(r_vals.sum())
        cum_r     = np.cumsum(r_vals)
        run_max   = np.maximum.accumulate(cum_r)
        dd        = cum_r - run_max
        max_dd_r  = float(dd.min())
    else:
        avg_r    = 0.0
        total_r  = 0.0
        max_dd_r = 0.0

    # Date range
    start_date = "?"
    end_date   = "?"
    if "open_time" in df.columns:
        try:
            dates      = pd.to_datetime(df["open_time"])
            start_date = str(dates.min().date())
            end_date   = str(dates.max().date())
        except Exception:
            pass

    return {
        "n":           n,
        "win_rate":    round(win_rate, 1),
        "pf":          round(pf, 2),
        "avg_r":       round(avg_r, 3),
        "total_r":     round(total_r, 2),
        "max_dd_r":    round(max_dd_r, 2),
        "net_pnl":     round(net_pnl, 2),
        "start_date":  start_date,
        "end_date":    end_date,
    }


# -----------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------
def build_report(stats: dict) -> str:
    n        = stats["n"]
    wr       = stats["win_rate"]
    pf       = stats["pf"]
    avg_r    = stats["avg_r"]
    total_r  = stats["total_r"]
    max_dd   = stats["max_dd_r"]
    net_pnl  = stats["net_pnl"]
    start    = stats["start_date"]
    end      = stats["end_date"]

    warn = ""
    if n < MIN_TRADES_FOR_WARN:
        warn = f"\n  NOTE: Stats comparison requires {MIN_TRADES_FOR_WARN}+ trades (currently {n}).\n"

    # Status markers
    wr_s  = status(wr,     BACKTEST_WIN_RATE_LOW, BACKTEST_WIN_RATE_HIGH)
    pf_s  = status(pf,     BACKTEST_PF_LOW, BACKTEST_PF_HIGH)
    avgr_s = status(avg_r, BACKTEST_AVG_R * 0.5, BACKTEST_AVG_R * 1.5)
    dd_s  = status(max_dd, BACKTEST_MAX_DD_R, 0.0)  # max_dd is negative, so lo is target, hi is 0

    line  = "=" * 55
    dline = "-" * 55

    r_sign = "+" if total_r >= 0 else ""
    ar_sign = "+" if avg_r >= 0 else ""

    report = (
        f"\n{line}\n"
        f"  LIVE PAPER TRADING PERFORMANCE REPORT\n"
        f"  Period: {start} to {end}\n"
        f"{line}\n"
        f"  Total Closed Trades : {n}\n"
        f"  Net P&L (USD)       : ${net_pnl:+.2f}\n"
        f"  Total Net R         : {r_sign}{total_r:.2f}R\n"
        f"{dline}\n"
        f"  {'Metric':<18} {'Live':>10}  {'Target':>15}  Status\n"
        f"{dline}\n"
        f"  {'Win Rate':<18} {wr:>9.1f}%  {'46% - 50%':>15}  {wr_s}\n"
        f"  {'Profit Factor':<18} {pf:>10.2f}  {'1.50 - 1.73':>15}  {pf_s}\n"
        f"  {'Avg R / trade':<18} {ar_sign}{avg_r:>9.3f}R  {'+0.39R':>15}  {avgr_s}\n"
        f"  {'Max Drawdown':<18} {max_dd:>10.1f}R  {'-6.0R max':>15}  {dd_s}\n"
        f"  {'Trades/month':<18} {n:>10}  {BACKTEST_TRADES_MONTH:>15}  (info)\n"
        f"{line}\n"
        f"{warn}"
    )

    return report


# -----------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------
def main():
    send_tg = "--telegram" in sys.argv

    print("\nLoading trade log...")
    df = load_closed_trades()

    if len(df) == 0:
        print("[INFO] No closed trades found yet. Start paper trading first.")
        sys.exit(0)

    stats  = calc_stats(df)
    report = build_report(stats)

    print(report)

    if send_tg:
        # Shorten for Telegram
        tg_msg = (
            f"[PERFORMANCE REPORT]\n"
            f"Period : {stats['start_date']} -> {stats['end_date']}\n"
            f"Trades : {stats['n']}\n"
            f"Win%   : {stats['win_rate']:.1f}%  (46-50%)\n"
            f"PF     : {stats['pf']:.2f}   (1.50-1.73)\n"
            f"Avg R  : {stats['avg_r']:+.3f}R (+0.39R)\n"
            f"Max DD : {stats['max_dd_r']:.1f}R (-6.0R)\n"
            f"Net R  : {stats['total_r']:+.2f}R\n"
            f"P&L    : ${stats['net_pnl']:+.2f}"
        )
        send_telegram(tg_msg)


if __name__ == "__main__":
    main()

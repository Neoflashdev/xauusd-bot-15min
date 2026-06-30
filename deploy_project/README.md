# XAUUSD M15 MSB Live Trading Bot

## What This Is

An automated paper trading bot for XAUUSD (Gold/USD) on the M15 timeframe.
Detects Market Structure Breaks (MSB) using a 3-model XGBoost ensemble.
Sends every trade alert and performance report to Telegram.

**Strategy (frozen — validated on 10 years of backtest data):**
- Long-only | 4H Bull Trend filter | Bullish MSB entry
- Ensemble Score threshold ≥ 0.65
- Risk:Reward = 1.5
- Stop loss below recent swing low

**Backtest Results (2016–2026):**
| Metric | Result |
|:---|:---|
| Win Rate | 46–50% |
| Profit Factor | 1.50–1.73 |
| Avg R / trade | +0.39R |
| Max Drawdown | -6R |

---

## Project Structure

```
xauusd-bot/
├── main.py                    # Live bot entry point
├── config.py                  # All settings loaded from .env
├── performance_tracker.py     # Compare live results vs backtest targets
├── train_models.py            # Train frozen XGBoost models (run once)
├── XAUUSD_Colab_Backtest.py  # Backtest & strategy library (do not edit)
├── requirements.txt           # pip dependencies
├── .env                       # Secrets and settings (DO NOT commit)
├── .gitignore
├── start_bot.bat              # One-click launcher
├── watchdog.bat               # Auto-restart launcher
├── logs/                      # live_bot.log, watchdog.log
│   └── (auto-created)
├── data/                      # state.json, live_trade_log.csv, live_decision_log.csv
│   └── (auto-created)
└── models/                    # XGBoost .pkl files (copy manually between servers)
    ├── expansion_model.pkl
    ├── fake_breakout_model.pkl
    ├── trend_continuation_model.pkl
    └── feature_metadata.pkl
```

---

## Installation (any Windows server)

### 1. Install Python 3.12+
Download from [python.org](https://python.org). During install, **check "Add Python to PATH"**.

### 2. Install MetaTrader 5
Download from your broker or [MetaQuotes](https://www.metatrader5.com/en/download).
Log in to your demo account. Keep credentials for the `.env` file.

### 3. Clone the repository

```cmd
git clone https://github.com/YOUR_USERNAME/xauusd-bot.git
cd xauusd-bot
```

### 4. Install Python dependencies

```cmd
pip install -r requirements.txt
```

### 5. Configure `.env`

Copy the `.env` template and fill in your credentials:

```env
MT5_LOGIN=5052263592
MT5_PASSWORD=YourPassword
MT5_SERVER=MetaQuotes-Demo

TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

EXECUTION_MODE=PAPER_MT5_DEMO
RISK_PCT=0.5
```

### 6. Copy model files

Model `.pkl` files are **not in Git** (too large). Copy the `models/` folder from your laptop to the server:

```
models/expansion_model.pkl
models/fake_breakout_model.pkl
models/trend_continuation_model.pkl
models/feature_metadata.pkl
```

> If you don't have model files yet, run `python train_models.py` on your laptop first.

### 7. Run the bot

```cmd
python main.py
```

Or double-click `start_bot.bat` for one-click launch.

For automatic restart on crash:

```cmd
watchdog.bat
```

---

## Monitoring

All trade alerts and performance reports go to your Telegram bot.

### Every trade OPEN:
```
[TRADE OPENED]
Bar    : 2026-06-26 14:00:00
Entry  : 2320.50
SL     : 2308.00
TP     : 2337.75
Lot    : 0.05
Risk   : 0.49%
Score  : 0.712
Ticket : 12345678
```

### Every trade CLOSE (with running stats):
```
[TRADE CLOSED] WIN
Ticket : 12345678
Entry  : 2320.50
Exit   : 2337.75
SL/TP  : 2308.00 / 2337.75
R      : +1.50R
P&L    : +$18.72

--- Running Stats (12 trades) ---
Win%   : 50.0%   Target 46-50%  [OK]
PF     : 1.68    Target 1.50-1.73 [OK]
Avg R  : +0.380R Target +0.39R [OK]
Max DD : -2.5R   Limit -6.0R  [OK]
```

### Manual performance report:
```cmd
python performance_tracker.py
python performance_tracker.py --telegram
```

---

## Switching Between Servers (Oracle → Google Cloud → Contabo)

1. Stop the bot on the old server.
2. On the new server:
   ```cmd
   git clone https://github.com/YOUR_USERNAME/xauusd-bot.git
   pip install -r requirements.txt
   ```
3. Copy these two things only:
   - `.env` (your credentials and settings)
   - `models/` folder (the .pkl files)
4. Optionally copy `data/` to continue tracking trade history.
5. Start the bot: `python main.py`

**That's it. Nothing else changes.**

---

## Training Models (first time only)

Models are trained once on your laptop and the `.pkl` files are copied to servers.
Never train inside the live bot.

```cmd
python train_models.py
```

This saves to `models/`.

---

## Key Rules

- **Never edit strategy constants** in `main.py` — change only `.env`
- **Never retrain models on the server** — only load frozen `.pkl` files
- **Never hardcode paths** — use `BASE / "folder"` pattern
- **Keep `.env` secret** — it contains your MT5 password and Telegram token

---

## Troubleshooting

| Problem | Fix |
|:---|:---|
| `MT5 init failed` | Check `.env` credentials. Make sure MT5 is installed. |
| `Model file missing` | Copy `models/` folder from laptop or run `train_models.py` |
| No Telegram messages | Check `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Bot closes immediately | Check `logs/live_bot.log` for error details |

@echo off
REM ============================================================
REM start_bot.bat
REM One-click launcher for XAUUSD M15 MSB Live Bot.
REM Double-click this file to start the bot.
REM ============================================================

title XAUUSD MSB Live Bot

echo.
echo  ============================================================
echo    XAUUSD M15 MSB BOT - STARTING
echo  ============================================================
echo.
echo  Mode   : PAPER_MT5_DEMO (set in .env)
echo  Log    : logs\live_bot.log
echo  Trades : data\live_trade_log.csv
echo.

REM Change to the bot directory (works even if called from elsewhere)
cd /d "%~dp0"

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.12+ and add to PATH.
    pause
    exit /b 1
)

REM Check main.py exists
if not exist "main.py" (
    echo [ERROR] main.py not found in %~dp0
    pause
    exit /b 1
)

REM Check .env exists
if not exist ".env" (
    echo [WARN] .env file not found. Copy .env and fill in credentials.
    echo        Continuing with environment variables only...
    echo.
)

REM Run the bot
echo  Starting main.py ...
echo.
python main.py

REM If we reach here the bot exited (not via watchdog)
echo.
echo [INFO] Bot exited. Check logs\live_bot.log for details.
echo.
pause

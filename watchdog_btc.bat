@echo off
REM ============================================================
REM watchdog_btc.bat
REM Auto-restarts the BTCUSD bot if it crashes or exits.
REM ============================================================

title BTCUSD Bot Watchdog

cd /d "%~dp0"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:loop
echo.
echo [%date% %time%] Watchdog: Starting BTCUSD bot... >> logs\watchdog_btc.log
echo [%date% %time%] Watchdog: Starting BTCUSD bot...

set ENV_FILE=.env.btcusdm
python main.py

echo.
echo [%date% %time%] Watchdog: Bot exited. Restarting in 30 seconds... >> logs\watchdog_btc.log
echo [%date% %time%] Watchdog: Bot exited. Restarting in 30 seconds...
echo  (Close this window to stop permanently)
echo.

timeout /t 30 /nobreak

goto loop

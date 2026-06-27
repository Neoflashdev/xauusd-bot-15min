@echo off
REM ============================================================
REM watchdog_xau.bat
REM Auto-restarts the XAUUSD bot if it crashes or exits.
REM ============================================================

title XAUUSD Bot Watchdog

cd /d "%~dp0"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:loop
echo.
echo [%date% %time%] Watchdog: Starting XAUUSD bot... >> logs\watchdog_xau.log
echo [%date% %time%] Watchdog: Starting XAUUSD bot...

set ENV_FILE=.env.xauusdm
python main.py

echo.
echo [%date% %time%] Watchdog: Bot exited. Restarting in 30 seconds... >> logs\watchdog_xau.log
echo [%date% %time%] Watchdog: Bot exited. Restarting in 30 seconds...
echo  (Close this window to stop permanently)
echo.

timeout /t 30 /nobreak

goto loop

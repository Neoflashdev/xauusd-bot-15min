@echo off
REM ============================================================
REM watchdog_eth.bat
REM Auto-restarts the ETHUSD bot if it crashes or exits.
REM ============================================================

title ETHUSD Bot Watchdog

cd /d "%~dp0"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:loop
echo.
echo [%date% %time%] Watchdog: Starting ETHUSD bot... >> logs\watchdog_eth.log
echo [%date% %time%] Watchdog: Starting ETHUSD bot...

set ENV_FILE=.env.ethusdm
python main.py

echo.
echo [%date% %time%] Watchdog: Bot exited. Restarting in 30 seconds... >> logs\watchdog_eth.log
echo [%date% %time%] Watchdog: Bot exited. Restarting in 30 seconds...
echo  (Close this window to stop permanently)
echo.

timeout /t 30 /nobreak

goto loop

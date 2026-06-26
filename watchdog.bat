@echo off
REM ============================================================
REM watchdog.bat
REM Auto-restarts the bot if it crashes or exits.
REM
REM Usage: Double-click watchdog.bat instead of start_bot.bat
REM        to get automatic crash recovery.
REM
REM The bot will restart after 30 seconds if it stops.
REM To permanently stop: close this window.
REM ============================================================

title XAUUSD Bot Watchdog

cd /d "%~dp0"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:loop
echo.
echo [%date% %time%] Watchdog: Starting bot... >> logs\watchdog.log
echo [%date% %time%] Watchdog: Starting bot...

python main.py

echo.
echo [%date% %time%] Watchdog: Bot exited. Restarting in 30 seconds... >> logs\watchdog.log
echo [%date% %time%] Watchdog: Bot exited. Restarting in 30 seconds...
echo  (Close this window to stop permanently)
echo.

timeout /t 30 /nobreak

goto loop

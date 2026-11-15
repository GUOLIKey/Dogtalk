@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
 echo ERROR: Place PawTalk_Web_Clean next to the original .venv, scripts, data and results.
 pause
 exit /b 1
)
"..\.venv\Scripts\python.exe" -B setup_live.py
if errorlevel 1 echo FAILED - send LIVE_SETUP_ERROR.txt. Do not send your API Key.
pause

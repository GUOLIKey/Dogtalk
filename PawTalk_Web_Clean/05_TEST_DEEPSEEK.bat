@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
 echo ERROR: Put PawTalk_Web_Clean directly inside the original project folder.
 echo It must be next to .venv, scripts and data. No old Web folder is needed.
 pause
 exit /b 1
)
"..\.venv\Scripts\python.exe" -B start_app.py --test
if errorlevel 1 echo FAILED - send the message above, without your API Key.
pause

@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist "..\.venv\Scripts\python.exe" (
 echo ERROR: Put these files INSIDE the original PawTalk_Web_Clean folder.
 echo The parent folder must contain .venv, scripts, data and results.
 pause
 exit /b 1
)
"..\.venv\Scripts\python.exe" -B -u scene_results_view.py
if errorlevel 1 echo STOPPED: Send the error or training log. Do not send your API key.
pause

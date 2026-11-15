@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
 echo Place PawTalk_Web_Clean next to the original project data, scripts and .venv folders.
 pause
 exit /b 1
)
"..\.venv\Scripts\python.exe" prepare_audio.py
if errorlevel 1 echo FAILED - please send the error shown above.
pause

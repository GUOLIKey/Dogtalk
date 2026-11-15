@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
 echo ERROR: Put PawTalk_Web_Clean next to the original .venv folder.
 pause
 exit /b 1
)
"..\.venv\Scripts\python.exe" -B setup_emotion.py
pause

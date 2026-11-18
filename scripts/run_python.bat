@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" %*
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 %*
    ) else (
        python %*
    )
)
exit /b %errorlevel%

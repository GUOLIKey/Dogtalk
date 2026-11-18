@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
echo =================================================
echo DogTalk - REAL 6-class retraining
echo =================================================
if not exist "%PY%" (
 echo [ERROR] Missing original .venv: %PY%
 goto :fail
)
if not exist "run_scene_experiment.py" (
 echo [ERROR] Missing six-scene training source: run_scene_experiment.py
 echo Download feature/six-scene-v1 source and merge it into this folder first.
 goto :fail
)
if not exist "scene_training.py" (
 echo [ERROR] Missing scene_training.py
 goto :fail
)
if not exist "dev_manifest.csv" (
 echo [ERROR] Missing dev_manifest.csv
 goto :fail
)
if not exist "%ROOT%\results\optimization_step1\splits\LOCKED_holdout_DO_NOT_TUNE.csv" (
 echo [ERROR] Missing original locked holdout list.
 echo Do not generate a replacement split.
 goto :fail
)
echo [INFO] Real DEV audio required; no emotion/LLM calls.
"%PY%" -B -u run_scene_experiment.py
if errorlevel 1 goto :fail
echo.
echo [SUCCESS] Check results\scene_six_v1\trained_run_*
echo Send summary.txt and metrics.json for review.
pause
exit /b 0
:fail
echo.
echo [STOPPED] Training did not complete.
echo Send the console error or results\scene_six_v1\train_*.log
pause
exit /b 1

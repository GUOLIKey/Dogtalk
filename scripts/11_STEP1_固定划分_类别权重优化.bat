@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HOME=%CD%\.hf_cache
set HF_HUB_CACHE=%CD%\.hf_cache\hub
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Existing project .venv not found. Place this patch into the existing PawTalk project root.
    goto :end
)
if not exist "results\train_manifest.csv" (
    echo [ERROR] Missing train manifest. Run 02_AUDIT_DATA.bat in your original project first.
    goto :end
)
if not exist "results\clap_embeddings_cache.sqlite3" (
    echo [ERROR] Missing CLAP embedding cache. Run 10_CLAP_SUPERVISED_FULL.bat first.
    goto :end
)
echo [STEP 1] Lock stratified holdout and compare class-weight policies on development CV.
echo [INFO] No full CLAP model inference. Reuses cached embeddings.
echo [INFO] Existing original reports are kept unchanged.
".venv\Scripts\python.exe" "scripts\step1_split_and_weights.py"
if errorlevel 1 (
    echo [FAILED] Step 1 encountered an error. Copy the last lines to ChatGPT.
) else (
    echo [PASS] Step 1 complete. See results\optimization_step1\STEP1_结果摘要.md
)
:end
pause

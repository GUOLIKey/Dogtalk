@echo off
setlocal
cd /d "%~dp0"
if not exist "DEEPSEEK_CONFIG.ini" copy "DEEPSEEK_CONFIG.example.ini" "DEEPSEEK_CONFIG.ini" >nul
notepad "DEEPSEEK_CONFIG.ini"

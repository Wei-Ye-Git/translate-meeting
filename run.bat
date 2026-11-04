@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHONUTF8=1"
    set "PYTHONIOENCODING=utf-8"
    set "HF_HUB_OFFLINE=1"
    set "TRANSFORMERS_OFFLINE=1"
    "%~dp0.venv\Scripts\python.exe" "%~dp0main.py" > "%~dp0run.log" 2>&1
) else (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

if errorlevel 1 (
    echo The translator stopped with an error. See run.log.
)
pause
endlocal

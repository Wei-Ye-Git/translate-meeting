@echo off
setlocal
cd /d "%~dp0"

rem Prefer the newest locally installed Python with compatible Windows wheels.
set "PYTHON="
py -3.14 --version >nul 2>&1 && set "PYTHON=py -3.14"
if not defined PYTHON py -3.11 --version >nul 2>&1 && set "PYTHON=py -3.11"

if not defined PYTHON (
    echo Python 3.14 or 3.11 was not found.
    echo Install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Using %PYTHON%
if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv ".venv"
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_OPTIONS="
"%VENV_PYTHON%" -m pip install --upgrade pip %PIP_OPTIONS%
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)
"%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt" %PIP_OPTIONS%
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo Downloading the local models. The first run may take a while.
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_DISABLE_TELEMETRY=1"
"%VENV_PYTHON%" "%~dp0download_models.py"
if errorlevel 1 (
    echo Failed to download models. Check the network and run setup.bat again.
    echo Model download uses hf-mirror.com instead of ModelScope.
    pause
    exit /b 1
)

echo Setup complete. Double-click run.bat to start the translator.
pause
endlocal

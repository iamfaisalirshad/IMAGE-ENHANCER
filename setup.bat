@echo off
REM ==================================================
REM  IMAGE ENHANCER - one-time setup for Windows
REM  Creates a virtual environment and installs deps.
REM ==================================================
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10-3.12 from python.org
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 goto :err

echo [2/3] Activating and upgrading pip...
call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip

echo [3/3] Installing requirements (this downloads, please wait)...
python -m pip install -r requirements.txt
if errorlevel 1 goto :err

echo.
echo Setup complete. Run the project with:  run.bat
echo.
pause
exit /b 0

:err
echo.
echo [ERROR] Setup failed. Make sure Python 3.10-3.12 is installed.
pause
exit /b 1
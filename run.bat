@echo off
REM ==================================================
REM  IMAGE ENHANCER - Windows launcher
REM  Double-click this file (or run: run.bat) to start.
REM ==================================================
setlocal
cd /d "%~dp0"

REM Use the project virtual environment if it exists.
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    REM Fall back to any python on PATH.
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python 3.10-3.12 first.
        pause
        exit /b 1
    )
    set "PY=python"
)

echo Starting IMAGE ENHANCER...
echo.
"%PY%" main.py
if errorlevel 1 (
    echo.
    echo [ERROR] main.py exited with an error.
    echo         Did you run:  pip install -r requirements.txt  ?
)
echo.
pause
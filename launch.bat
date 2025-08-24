# launch.bat - Windows Batch File
@echo off
echo ========================================
echo   Multifactor Stock Analysis System
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

:: Check if required directories exist
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "output" mkdir output
if not exist "config" mkdir config

:: Start the backend server (if you have one)
echo Starting backend server...
start /b python src/backend/app.py >logs/backend.log 2>&1

:: Wait for backend to initialize
timeout /t 3 /nobreak >nul

:: Open the UI in default browser
echo Launching UI in browser...
start "" "ui/dashboard.html"

echo.
echo System is running!
echo Press Ctrl+C to stop the backend server
echo.

:: Keep the window open and show backend logs
python src/backend/app.py
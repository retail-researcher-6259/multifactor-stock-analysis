@echo off
REM MSAS System Launcher
REM This batch file activates the virtual environment and runs the MSAS UI

echo Starting MSAS System...
echo.

REM Activate the virtual environment
echo Activating virtual environment...
call "C:\Miscellaneous_Programs\Python311\riskfolio_env\Scripts\activate.bat"

REM Check if activation was successful
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated successfully
echo.

REM Run the MSAS UI script
echo Launching MSAS automation...
python integrated_daemon.py

REM Check if the script ran successfully
if errorlevel 1 (
    echo MSAS UI encountered an error
    pause
    exit /b 1
)

echo MSAS UI has been closed
pause
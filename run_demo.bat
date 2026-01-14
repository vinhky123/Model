@echo off
REM Quick start script for demo (Windows)

echo ==================================
echo Time Series Forecasting Demo
echo ==================================
echo.

REM Check if demo directory exists
if not exist "demo" (
    echo Error: demo directory not found!
    pause
    exit /b 1
)

REM Check if params directory exists
if not exist "params" (
    echo Creating params directory...
    mkdir params
    
    REM Try to copy from checkpoints
    if exist "checkpoints" (
        echo Copying checkpoints to params...
        xcopy /E /I /Q checkpoints params 2>nul
    )
)

REM Check if dataset exists
if not exist "dataset\ETT-small\ETTm2.csv" (
    echo Warning: ETTm2.csv not found!
    echo Please make sure dataset is in: dataset\ETT-small\ETTm2.csv
    echo.
)

REM Install dependencies
echo Installing dependencies...
cd demo


if errorlevel 1 (
    echo.
    echo Failed to install dependencies!
    echo.
    pause
    exit /b 1
)

echo.
echo Setup complete!
echo.
echo Starting demo server...
echo Open browser at: http://localhost:5000
echo.
echo Press Ctrl+C to stop server
echo.

REM Run app
python app.py

REM Pause if error occurs
if errorlevel 1 (
    echo.
    echo Server crashed! Check error above.
    pause
)


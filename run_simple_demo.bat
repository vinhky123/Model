@echo off
REM Quick run simple demo

echo ========================================
echo Simple Demo - Time Series Forecasting
echo ========================================
echo.

REM Check if matplotlib is installed
python -c "import matplotlib" 2>nul
if errorlevel 1 (
    echo Installing matplotlib...
    pip install matplotlib
    if errorlevel 1 (
        echo Failed to install matplotlib!
        pause
        exit /b 1
    )
)

echo Running demo...
echo.

REM Run with default settings
python simple_demo.py

echo.
pause


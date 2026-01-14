@echo off
REM Debug script - Check all requirements and show errors

echo ========================================
echo Demo Debug - Checking Requirements
echo ========================================
echo.

REM 1. Check Python
echo [1] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo    OK - Python found
echo.

REM 2. Check pip
echo [2] Checking pip...
pip --version
if errorlevel 1 (
    echo ERROR: pip not found!
    pause
    exit /b 1
)
echo    OK - pip found
echo.

REM 3. Check demo directory
echo [3] Checking demo directory...
if exist "demo" (
    echo    OK - demo directory exists
) else (
    echo ERROR: demo directory not found!
    echo Please run this from project root directory
    pause
    exit /b 1
)
echo.

REM 4. Check demo files
echo [4] Checking demo files...
if exist "demo\app.py" (
    echo    OK - app.py found
) else (
    echo ERROR: demo\app.py not found!
    pause
    exit /b 1
)

if exist "demo\requirements.txt" (
    echo    OK - requirements.txt found
) else (
    echo ERROR: demo\requirements.txt not found!
    pause
    exit /b 1
)

if exist "demo\templates\index.html" (
    echo    OK - templates\index.html found
) else (
    echo ERROR: demo\templates\index.html not found!
    pause
    exit /b 1
)
echo.

REM 5. Check params directory
echo [5] Checking params directory...
if exist "params" (
    echo    OK - params directory exists
    dir /b params | find /c /v "" > temp.txt
    set /p count=<temp.txt
    del temp.txt
    echo    Found checkpoints (check manually if empty)
) else (
    echo WARNING: params directory not found!
    echo Will create it, but you need to add model checkpoints
)
echo.

REM 6. Check dataset
echo [6] Checking dataset...
if exist "dataset\ETT-small\ETTm2.csv" (
    echo    OK - ETTm2.csv found
) else (
    echo WARNING: dataset\ETT-small\ETTm2.csv not found!
    echo Demo may not work without dataset
)
echo.

REM 7. Check parent directory modules
echo [7] Checking parent modules...
if exist "data_provider" (
    echo    OK - data_provider found
) else (
    echo ERROR: data_provider directory not found!
    echo Make sure you're in the project root
    pause
    exit /b 1
)

if exist "models" (
    echo    OK - models found
) else (
    echo ERROR: models directory not found!
    pause
    exit /b 1
)

if exist "exp" (
    echo    OK - exp found
) else (
    echo ERROR: exp directory not found!
    pause
    exit /b 1
)
echo.

REM 8. Try installing dependencies
echo [8] Installing dependencies...
cd demo
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    echo Check error messages above
    cd ..
    pause
    exit /b 1
)
cd ..
echo    OK - Dependencies installed
echo.

REM 9. Test Python imports
echo [9] Testing Python imports...
python -c "import flask; print('   OK - Flask imported')"
if errorlevel 1 (
    echo ERROR: Cannot import Flask
    pause
    exit /b 1
)

python -c "import torch; print('   OK - PyTorch imported')"
if errorlevel 1 (
    echo ERROR: Cannot import PyTorch
    pause
    exit /b 1
)

python -c "import numpy; print('   OK - NumPy imported')"
if errorlevel 1 (
    echo ERROR: Cannot import NumPy
    pause
    exit /b 1
)

python -c "import pandas; print('   OK - Pandas imported')"
if errorlevel 1 (
    echo ERROR: Cannot import Pandas
    pause
    exit /b 1
)
echo.

echo ========================================
echo All checks passed!
echo ========================================
echo.
echo You can now run: run_demo.bat
echo Or manually: cd demo ^&^& python app.py
echo.
pause


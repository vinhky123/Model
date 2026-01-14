@echo off
REM Find best samples with lowest MSE

echo ========================================
echo Find Best Samples
echo ========================================
echo.

REM Check arguments
if "%1"=="" (
    echo Usage: find_best_samples.bat [model] [dataset] [top_k]
    echo.
    echo Examples:
    echo   find_best_samples.bat TimeStar ETTm2 10
    echo   find_best_samples.bat TimeXer ETTh1 20
    echo.
    echo Running with defaults: TimeStar ETTm2 10
    echo.
    set MODEL=TimeStar
    set DATA=ETTm2
    set TOP_K=10
) else (
    set MODEL=%1
    set DATA=%2
    if "%3"=="" (
        set TOP_K=10
    ) else (
        set TOP_K=%3
    )
)

echo Model: %MODEL%
echo Dataset: %DATA%
echo Top K: %TOP_K%
echo.

python simple_demo.py --model %MODEL% --data %DATA% --find_best --top_k %TOP_K%

echo.
pause


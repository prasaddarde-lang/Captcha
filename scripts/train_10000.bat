@echo off
REM ==========================================================================
REM  Build the 10,000-sample synthetic dataset, validate it, train and compare
REM  model architectures, and write the performance report.
REM  Usage: scripts\train_10000.bat [--force]
REM    --force  regenerate the dataset and retrain even if nothing changed
REM  Exit: 0 ok, 1 failed, 3 paused by a resource/time limit (re-run to resume)
REM ==========================================================================
setlocal EnableExtensions
chcp 65001 >nul
call "%~dp0_python.bat" || exit /b 99
cd /d "%~dp0.."

echo === [1/4] Generating dataset (10,000 samples) ===
"%VPY%" training\generate_dataset.py --count 10000 %1
if errorlevel 1 goto :fail

echo === [2/4] Validating every sample ===
"%VPY%" training\dataset_validator.py
if errorlevel 1 goto :fail

echo === [3/4] Training and comparing architectures ===
"%VPY%" training\train_classifier.py %1
set "RC=%ERRORLEVEL%"
if "%RC%"=="3" (
    echo Training paused by a resource limit - run this script again to resume.
    endlocal & exit /b 3
)
if not "%RC%"=="0" goto :fail

echo === [4/4] Writing performance report ===
"%VPY%" training\training_report.py
echo.
echo Done. Report: reports\performance\training_report_latest.md
echo A named reviewer puts a candidate into production with:
echo     scripts\approve_model.bat ^<version^> "Your Name"
endlocal & exit /b 0

:fail
echo FAILED - see the messages above.
endlocal & exit /b 1

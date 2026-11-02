@echo off
REM ==========================================================================
REM  Evaluate a model on the fixed test set (or another split).
REM  Usage: scripts\evaluate_model.bat                 (production model)
REM         scripts\evaluate_model.bat img-1.0.2       (a registered version)
REM         scripts\evaluate_model.bat img-1.0.2 validation
REM ==========================================================================
setlocal EnableExtensions
chcp 65001 >nul
call "%~dp0_python.bat" || exit /b 99
cd /d "%~dp0.."
set "SPLIT=test"
if not "%~2"=="" set "SPLIT=%~2"
if "%~1"=="" (
    "%VPY%" training\evaluate_model.py --production --split %SPLIT%
) else if /I "%~1"=="production" (
    "%VPY%" training\evaluate_model.py --production --split %SPLIT%
) else (
    "%VPY%" training\evaluate_model.py --version %~1 --split %SPLIT%
)
set "RC=%ERRORLEVEL%"
"%VPY%" training\training_report.py >nul
endlocal & exit /b %RC%

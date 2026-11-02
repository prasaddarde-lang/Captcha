@echo off
REM ==========================================================================
REM  A named reviewer approves a candidate image model and promotes it to
REM  production. The previous production model is archived for rollback, and
REM  the new one is re-checked on the test set (automatic rollback if it fails).
REM  Usage: scripts\approve_model.bat img-1.0.0 "Jane Doe"
REM ==========================================================================
setlocal EnableExtensions
chcp 65001 >nul
if "%~2"=="" (
    echo Usage: scripts\approve_model.bat ^<version^> "Reviewer Name"
    endlocal & exit /b 40
)
call "%~dp0_python.bat" || exit /b 99
cd /d "%~dp0.."
"%VPY%" training\model_registry.py approve %~1 --approver "%~2"
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" "%VPY%" training\training_report.py >nul
endlocal & exit /b %RC%

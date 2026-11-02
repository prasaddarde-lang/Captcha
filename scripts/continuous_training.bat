@echo off
REM ==========================================================================
REM  Bounded continuous-improvement run (resumable, single instance).
REM  Usage: scripts\continuous_training.bat [max_iterations] [max_minutes]
REM         scripts\continuous_training.bat status
REM  Defaults come from training\training_config.yaml (continuous section).
REM  Stops at the iteration/time limit, the augmentation budget, low disk,
REM  or after N iterations without a validated improvement (patience).
REM  Candidates that meet the criteria are only RECOMMENDED: a named reviewer
REM  approves them with scripts\approve_model.bat.
REM  Exit: 0 ok, 4 another controller is running, 5 resource limit
REM ==========================================================================
setlocal EnableExtensions
chcp 65001 >nul
call "%~dp0_python.bat" || exit /b 99
cd /d "%~dp0.."
if /I "%~1"=="status" (
    "%VPY%" training\continuous_trainer.py status
    goto :done
)
set "ARGS="
if not "%~1"=="" set ARGS=--max-iterations %~1
if not "%~2"=="" set ARGS=%ARGS% --max-minutes %~2
"%VPY%" training\continuous_trainer.py run %ARGS%
:done
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%

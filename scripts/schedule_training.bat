@echo off
REM ==========================================================================
REM  Register (or remove) a DAILY Windows scheduled task that runs one bounded
REM  continuous-training run (3 iterations, 30 minutes). Run it yourself - it
REM  changes Task Scheduler for your user account.
REM  Usage: scripts\schedule_training.bat 02:00      (create / update)
REM         scripts\schedule_training.bat /delete    (remove)
REM  Note: Task Scheduler limits the command to 261 characters; if the project
REM  folder path is very long, move the project to a shorter path first.
REM ==========================================================================
setlocal EnableExtensions
set "TASK=CaptchaFW Continuous Training"
if /I "%~1"=="/delete" (
    schtasks /Delete /TN "%TASK%" /F
    endlocal & exit /b %ERRORLEVEL%
)
set "AT=%~1"
if "%AT%"=="" set "AT=02:00"
schtasks /Create /TN "%TASK%" /SC DAILY /ST %AT% /F /TR "\"%~dp0continuous_training.bat\" 3 30"
endlocal & exit /b %ERRORLEVEL%

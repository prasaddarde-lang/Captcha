@echo off
REM ==========================================================================
REM  Universal CAPTCHA Management Framework - RPA launcher
REM
REM  run.bat <input.json> [output.json] [platform]
REM      platform: generic (default) | uipath | automation_anywhere | blue_prism | power_automate
REM  type task.json | run.bat
REM  run.bat --input-json "{...}" --timeout 60          (any main.py arguments are passed through)
REM  run.bat operator list | operator complete <id> --operator "Name" | serve | report
REM
REM  stdout    : JSON result (key=value lines for blue_prism)
REM  exit code : 0 ok/verified, 10 detected, 11 human verification required, 12 pending,
REM              20 test failed, 21 rejected, 22 expired, 30 timeout, 40 invalid input,
REM              41 policy blocked, 50 network error, 99 error
REM ==========================================================================
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"

set "VPY="
if defined CAPTCHA_FW_PYTHON set "VPY=%CAPTCHA_FW_PYTHON%"
if not defined VPY if defined CAPTCHA_FW_VENV if exist "%CAPTCHA_FW_VENV%\Scripts\python.exe" set "VPY=%CAPTCHA_FW_VENV%\Scripts\python.exe"
if not defined VPY if exist "%ROOT%.venv\Scripts\python.exe" set "VPY=%ROOT%.venv\Scripts\python.exe"
if not defined VPY (
    where python >nul 2>&1 || (
        echo {"task_id":"UNKNOWN","status":"ERROR","message":"Python not found. Run install.bat first.","exit_code":99}
        endlocal & exit /b 99
    )
    set "VPY=python"
)

REM No arguments: read JSON from stdin
if "%~1"=="" (
    "%VPY%" "%ROOT%main.py" run
    goto :done
)

REM Pass-through for flags and sub-commands
set "FIRST=%~1"
if "%FIRST:~0,1%"=="-" goto :passthru
for %%C in (run operator dataset model feedback report serve) do if /I "%~1"=="%%C" goto :passthru

if not exist "%~1" (
    echo {"task_id":"UNKNOWN","status":"INVALID_INPUT","message":"Input file not found: %~nx1","exit_code":40}
    endlocal & exit /b 40
)
set "OUTARG="
if not "%~2"=="" set OUTARG=--output-file "%~2"
set "PLATARG="
if not "%~3"=="" set PLATARG=--platform "%~3"
"%VPY%" "%ROOT%main.py" run --input-file "%~1" %OUTARG% %PLATARG%
goto :done

:passthru
"%VPY%" "%ROOT%main.py" %*

:done
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%

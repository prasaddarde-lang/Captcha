@echo off
REM Sets VPY to the framework's Python (same lookup order as run.bat). Called by the other scripts.
set "VPY="
if defined CAPTCHA_FW_PYTHON set "VPY=%CAPTCHA_FW_PYTHON%"
if not defined VPY if defined CAPTCHA_FW_VENV if exist "%CAPTCHA_FW_VENV%\Scripts\python.exe" set "VPY=%CAPTCHA_FW_VENV%\Scripts\python.exe"
if not defined VPY if exist "%~dp0..\.venv\Scripts\python.exe" set "VPY=%~dp0..\.venv\Scripts\python.exe"
if not defined VPY (
    where python >nul 2>&1 || (echo ERROR: Python not found. Run install.bat first. 1>&2 & exit /b 99)
    set "VPY=python"
)
set "PYTHONIOENCODING=utf-8"
exit /b 0

@echo off
REM ==========================================================================
REM  Restore a previously approved image model.
REM  Usage: scripts\rollback_model.bat            (previous production model)
REM         scripts\rollback_model.bat img-1.0.0
REM ==========================================================================
setlocal EnableExtensions
call "%~dp0_python.bat" || exit /b 99
cd /d "%~dp0.."
"%VPY%" training\model_registry.py rollback %~1
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%

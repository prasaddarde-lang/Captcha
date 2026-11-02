@echo off
REM ==========================================================================
REM  Universal CAPTCHA Management Framework - installer
REM  Usage:  install.bat [/nobrowser] [/notest]
REM    /nobrowser  skip "playwright install chromium" (HTML-only detection still works)
REM    /notest     skip the post-install test suite
REM ==========================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul

set "SKIP_BROWSER=0"
set "SKIP_TEST=0"
for %%A in (%*) do (
    if /I "%%~A"=="/nobrowser" set "SKIP_BROWSER=1"
    if /I "%%~A"=="/notest" set "SKIP_TEST=1"
)

set "VENV_DIR=%~dp0.venv"
if defined CAPTCHA_FW_VENV set "VENV_DIR=%CAPTCHA_FW_VENV%"
set "RESULT_PY=FAIL" & set "RESULT_VENV=FAIL" & set "RESULT_DEPS=FAIL" & set "RESULT_BROWSER=SKIPPED" & set "RESULT_VERIFY=FAIL" & set "RESULT_TESTS=SKIPPED"

echo.
echo === [1/6] Checking Python 3.12+ ===
set "PY="
where py >nul 2>&1 && (py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1 && set "PY=py -3")
if not defined PY (
    where python >nul 2>&1 && (python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1 && set "PY=python")
)
if not defined PY (
    echo ERROR: Python 3.12 or newer was not found on PATH.
    echo        Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    goto :summary
)
for /f "delims=" %%V in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%V"
echo Found Python !PYVER! (%PY%)
set "RESULT_PY=OK (!PYVER!)"

echo.
echo === [2/6] Creating virtual environment: "%VENV_DIR%" ===
REM Deep folder paths can exceed the Windows 260-character limit during pip installs.
set "P=%VENV_DIR%"
set /a LEN=0
:len_loop
if defined P (set "P=!P:~1!" & set /a LEN+=1 & goto :len_loop)
if !LEN! GTR 90 (
    echo WARNING: venv path is !LEN! characters long. If pip fails with "No such file or directory",
    echo          enable Windows long paths or set CAPTCHA_FW_VENV to a short folder, e.g. C:\venvs\captcha_fw
)
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment already exists - reusing it.
) else (
    %PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment.
        goto :summary
    )
)
set "VPY=%VENV_DIR%\Scripts\python.exe"
set "RESULT_VENV=OK"

echo.
echo === [3/6] Installing dependencies ===
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check -q
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 (
    echo ERROR: dependency installation failed. Check proxy / network settings
    echo        ^(set HTTPS_PROXY if you are behind a corporate proxy^).
    goto :summary
)
set "RESULT_DEPS=OK"

echo.
echo === [4/6] Installing Playwright browser ===
if "%SKIP_BROWSER%"=="1" (
    echo Skipped by /nobrowser.
) else (
    "%VPY%" -m playwright install chromium
    if errorlevel 1 (
        echo WARNING: Playwright browser download failed. HTML/JSON detection still works;
        echo          live-browser features need: "%VPY%" -m playwright install chromium
        set "RESULT_BROWSER=WARNING"
    ) else (
        set "RESULT_BROWSER=OK"
    )
)

echo.
echo === [5/6] Verifying installation ===
"%VPY%" -c "import pydantic, fastapi, httpx, cv2, numpy, playwright; print('imports OK: pydantic', pydantic.VERSION, '| opencv', cv2.__version__)"
if errorlevel 1 goto :summary
if not exist sites.json copy /Y sites.example.json sites.json >nul
"%VPY%" main.py dataset generate --per-label 10 >nul
"%VPY%" main.py run --input-file samples\sample_input_test.json --output-file logs\install_check.json >nul
if errorlevel 1 (
    echo ERROR: sample task failed - see logs\install_check.json and logs\captcha_framework.log
    goto :summary
)
echo Sample task OK - result written to logs\install_check.json
set "RESULT_VERIFY=OK"

echo.
echo === [6/6] Running test suite ===
if "%SKIP_TEST%"=="1" (
    echo Skipped by /notest.
) else (
    "%VPY%" -m pytest -q tests
    if errorlevel 1 (set "RESULT_TESTS=FAILED") else (set "RESULT_TESTS=OK")
)

:summary
echo.
echo ==================== SETUP RESULTS ====================
echo  Python             : %RESULT_PY%
echo  Virtual environment: %RESULT_VENV%
echo  Dependencies       : %RESULT_DEPS%
echo  Playwright browser : %RESULT_BROWSER%
echo  Verification       : %RESULT_VERIFY%
echo  Tests              : %RESULT_TESTS%
echo =======================================================
if "%RESULT_VERIFY%"=="OK" (
    echo Ready. Try:  run.bat samples\sample_input_test.json
    endlocal & exit /b 0
)
echo Setup did not complete. See messages above and README.md ^(Troubleshooting^).
endlocal & exit /b 1

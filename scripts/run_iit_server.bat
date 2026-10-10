@echo off
REM ==============================================================================
REM FlowEdit Enterprise TTS Service — Windows / Local Launcher
REM ==============================================================================

set PORT=%1
if "%PORT%"=="" set PORT=8000

set HOST=%2
if "%HOST%"=="" set HOST=0.0.0.0

cd /d "%~dp0\.."
set PYTHONPATH=%CD%;%PYTHONPATH%
set FLOWEDIT_HOST=%HOST%
set FLOWEDIT_PORT=%PORT%

echo ======================================================================
echo    Starting FlowEdit Executive TTS Service on Windows / Server
echo ======================================================================
echo   Host: %HOST%
echo   Port: %PORT%
echo.
echo   Web UI: http://localhost:%PORT%/ui
echo ======================================================================
echo.

python -m uvicorn flowedit.api.main:app --host %HOST% --port %PORT% --workers 1
pause

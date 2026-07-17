@echo off
cd /d "%~dp0"

if not exist "app\main.py" (
  echo.
  echo ERROR: Wrong folder. This script must live in automl-platform.
  echo Current folder: %CD%
  echo.
  pause
  exit /b 1
)

echo Stopping any old server on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
  taskkill /F /PID %%a >nul 2>&1
)

echo.
echo Starting AutoMLOps Platform from:
echo   %CD%
echo.
echo   Web UI:   http://localhost:8000/ui/
echo   API docs: http://localhost:8000/docs
echo.
echo Wait for: "Web UI enabled at /ui/" in the log below.
echo.

py -3.11 -m uvicorn app.main:app --host 127.0.0.1 --port 8000

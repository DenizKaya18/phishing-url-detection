@echo off
REM Windows batch file to run Classical ML Pipeline

echo ========================================
echo Classical ML Pipeline Runner
echo ========================================
echo.

REM Check if virtual environment is activated
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please activate your virtual environment first
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "run.py" (
    echo ERROR: run.py not found
    echo Please run this script from src/classical_ml directory
    pause
    exit /b 1
)

echo Running Classical ML Pipeline...
echo.

python run.py

echo.
echo ========================================
echo Pipeline execution completed
echo ========================================
pause

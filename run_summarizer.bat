@echo off
echo Starting Web Summarizer...
echo.

REM Check if we're in the right directory
if not exist "app.py" (
    echo Error: app.py not found in current directory
    echo Please run this batch file from the web summarizer directory
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found
    echo Please make sure the virtual environment is set up in the 'venv' folder
    pause
    exit /b 1
)

REM Check if venv Python exists
if not exist "venv\Scripts\python.exe" (
    echo Error: Python executable not found in virtual environment
    echo Please make sure the virtual environment is properly set up
    pause
    exit /b 1
)

echo Virtual environment found.
echo.

REM Check Python version
venv\Scripts\python.exe --version
if errorlevel 1 (
    echo Error: Python in virtual environment is not working
    pause
    exit /b 1
)

echo.
echo Starting Streamlit application...
echo.
echo NOTE: Loading AI models may take several minutes on first run.
echo Please wait - you will see progress messages as models load...
echo.
echo The web summarizer will be available at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Start the Streamlit application
venv\Scripts\python.exe -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo ERROR: Flask application failed to start
    echo Check the error messages above for details
    pause
    exit /b 1
)

echo.
echo Application stopped.
pause 
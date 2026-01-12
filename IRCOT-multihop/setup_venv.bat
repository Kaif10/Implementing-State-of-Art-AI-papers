@echo off
REM Batch script to set up virtual environment for IRCOT
REM Run this from the IRCOT-multihop directory

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete! Virtual environment is active.
echo To activate in the future, run: venv\Scripts\activate.bat
pause

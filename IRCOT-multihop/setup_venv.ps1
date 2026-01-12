# PowerShell script to set up virtual environment for IRCOT
# Run this from the IRCOT-multihop directory

Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..." -ForegroundColor Green
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "`nSetup complete! Virtual environment is active." -ForegroundColor Green
Write-Host "To activate in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow

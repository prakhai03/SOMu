Write-Host "Setting up SOMu project..." -ForegroundColor Green

Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
python -m venv backend\venv

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
venv\Scripts\activate.bat

Write-Host "Installing backend dependencies..." -ForegroundColor Cyan
pip install -r requirements-windows.txt

Write-Host "Setting up React frontend..." -ForegroundColor Cyan
npx create-react-app frontend --template typescript

Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
Set-Location frontend
npm install @mui/material @emotion/react @emotion/styled @mui/icons-material
npm install axios d3 react-dropzone
Set-Location ..

Write-Host "`nSetup complete! To start the application:" -ForegroundColor Green
Write-Host "1. Start the backend server:" -ForegroundColor Yellow
Write-Host "   cd C:\Users\prakh\OneDrive\Desktop\SOMu\backend"
Write-Host "   venv\Scripts\activate.bat"
Write-Host "   python -m uvicorn main:app --reload"
Write-Host "`n2. Start the frontend development server:" -ForegroundColor Yellow
Write-Host "   cd frontend"
Write-Host "   npm start"
Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 
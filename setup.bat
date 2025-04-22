@echo off
echo Setting up SOMu project...

echo Creating Python virtual environment...
python -m venv backend\venv

echo Activating virtual environment...
call backend\venv\Scripts\activate.bat

echo Installing build dependencies...
python -m pip install --upgrade pip setuptools wheel

echo Installing backend dependencies...
pip install -r backend\requirements-windows.txt

echo Setting up React frontend...
npx create-react-app frontend --template typescript

echo Installing frontend dependencies...
cd frontend
call npm install @mui/material @emotion/react @emotion/styled @mui/icons-material
call npm install axios d3 react-dropzone
cd ..

echo.
echo Setup complete! To start the application:
echo 1. Start the backend server:
echo    cd backend
echo    call venv\Scripts\activate.bat
echo    python -m uvicorn main:app --reload
echo.
echo 2. Start the frontend development server:
echo    cd frontend
echo    npm start
echo.
pause 
@echo off
echo Starting SOMu application...

:: Start backend
cd backend
echo Installing backend dependencies...
pip install -r requirements.txt
echo Starting backend server...
start cmd /k python -m uvicorn main:app --reload

:: Start frontend
cd ../frontend
echo Installing frontend dependencies...
call npm install
echo Starting frontend development server...
start cmd /k npm start

echo SOMu application is starting...
echo Backend will be available at http://localhost:8000
echo Frontend will be available at http://localhost:3000
pause 
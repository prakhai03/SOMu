@echo off
echo Starting SOMu backend server...

cd backend
call venv\Scripts\activate.bat
python -m uvicorn main:app --reload

pause 
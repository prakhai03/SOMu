@echo off
echo Activating virtual environment...
call backend\venv\Scripts\activate.bat

echo Installing requirements...
pip install -r backend\requirements-windows.txt

echo Starting backend server...
cd backend
python -m uvicorn main:app --reload

pause 
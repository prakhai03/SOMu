@echo off
echo Setting up SOMu frontend...

cd frontend

echo Installing dependencies...
call npm install

echo Starting development server...
call npm start

pause 
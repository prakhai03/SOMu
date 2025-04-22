@echo off
echo Creating new React frontend...

echo Creating React app with TypeScript template...
call npx --yes create-react-app@latest frontend --template typescript
if errorlevel 1 (
    echo Failed to create React app
    pause
    exit /b 1
)

cd frontend
if errorlevel 1 (
    echo Failed to change directory to frontend
    pause
    exit /b 1
)

echo Installing Material-UI dependencies...
call npm install @mui/material @emotion/react @emotion/styled @mui/icons-material
if errorlevel 1 (
    echo Failed to install Material-UI dependencies
    pause
    exit /b 1
)

echo Installing additional dependencies...
call npm install axios d3 react-dropzone
if errorlevel 1 (
    echo Failed to install additional dependencies
    pause
    exit /b 1
)

echo.
echo Setup complete! To start the frontend server, run:
echo cd frontend
echo npm start
echo.

pause 
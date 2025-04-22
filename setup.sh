#!/bin/bash

# Create virtual environment for backend
echo "Setting up Python virtual environment..."
python -m venv backend/venv
source backend/venv/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create React app
echo "Setting up React frontend..."
npx create-react-app frontend --template typescript

# Install frontend dependencies
cd frontend
npm install @mui/material @emotion/react @emotion/styled @mui/icons-material
npm install axios d3 react-dropzone
cd ..

echo "Setup complete! To start the application:"
echo "1. Start the backend server:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   uvicorn main:app --reload"
echo ""
echo "2. Start the frontend development server:"
echo "   cd frontend"
echo "   npm start" 
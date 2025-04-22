from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from typing import List, Optional
from som_utils import SOMAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_data(
    file: UploadFile = File(...),
    selected_columns: str = Form(...),
    target_variable: Optional[str] = Form(None),
    algorithm: str = Form(...),
    hyperparameters: str = Form(...)
):
    try:
        logger.info(f"Received analysis request for file: {file.filename}")
        logger.info(f"Selected columns: {selected_columns}")
        logger.info(f"Target variable: {target_variable}")
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Hyperparameters: {hyperparameters}")
        
        # Read the CSV file
        logger.info("Reading CSV file...")
        df = pd.read_csv(file.file)
        logger.info(f"File read successfully. Shape: {df.shape}")
        
        # Parse parameters
        logger.info("Parsing parameters...")
        selected_columns = json.loads(selected_columns)
        hyperparameters = json.loads(hyperparameters)
        logger.info(f"Parsed selected columns: {selected_columns}")
        logger.info(f"Parsed hyperparameters: {hyperparameters}")
        
        # Select relevant columns
        if target_variable:
            logger.info(f"Filtering columns with target variable: {target_variable}")
            analysis_columns = [col for col in selected_columns if col != target_variable]
            df_analysis = df[analysis_columns + [target_variable]]
        else:
            logger.info("No target variable, using all selected columns")
            df_analysis = df[selected_columns]
        
        logger.info(f"Analysis data shape: {df_analysis.shape}")
        
        # Initialize and run SOM analysis
        logger.info("Initializing SOM analyzer...")
        analyzer = SOMAnalyzer(
            data=df_analysis,
            target_variable=target_variable,
            algorithm=algorithm,
            **hyperparameters
        )
        
        # Get analysis results
        logger.info("Running analysis...")
        results = analyzer.analyze()
        logger.info("Analysis completed successfully")
        
        return {
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/")
async def root():
    return {"message": "SOMu API is running"} 
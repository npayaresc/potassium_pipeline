#!/usr/bin/env python3
"""
Cloud-Agnostic REST API for Nitrogen Pipeline

This FastAPI server provides RESTful endpoints for training models and making predictions.
Designed to be cloud-agnostic and easily deployable on any cloud platform.
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Import pipeline components
from src.config.pipeline_config import Config
from src.models.predictor import Predictor
from main import (
    setup_pipeline_config,
    run_training_pipeline, 
    run_autogluon_pipeline, 
    run_tuning_pipeline,
    run_single_prediction_pipeline,
    run_batch_prediction_pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Nitrogen Prediction Pipeline API",
    description="Cloud-agnostic API for magnesium concentration prediction from spectral data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global configuration
config: Optional[Config] = None
predictor: Optional[Predictor] = None

# Pydantic models for API requests/responses
class TrainingRequest(BaseModel):
    """Request model for training pipelines."""
    use_gpu: bool = False
    pipeline_type: str = "standard"  # "standard", "autogluon", "tune"
    
class TrainingResponse(BaseModel):
    """Response model for training requests."""
    job_id: str
    status: str
    message: str
    timestamp: str

class PredictionRequest(BaseModel):
    """Request model for single predictions."""
    model_path: str
    
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_magnesium: float
    model_used: str
    timestamp: str
    sample_id: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    model_path: str
    reference_file: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    job_id: str
    total_samples: int
    successful_predictions: int
    failed_predictions: int
    output_file: str
    metrics: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    gpu_available: bool
    models_available: List[str]

# Global state for background jobs
background_jobs: Dict[str, Dict[str, Any]] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global config, predictor
    
    try:
        # Setup configuration
        config = setup_pipeline_config()
        predictor = Predictor(config)
        logger.info("Nitrogen Pipeline API started successfully")
        
        # Check GPU availability
        gpu_available = check_gpu_availability()
        logger.info(f"GPU available: {gpu_available}")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

def check_gpu_availability() -> bool:
    """Check if GPU is available for training."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_available_models() -> List[str]:
    """Get list of available trained models."""
    model_dir = Path(config.model_dir)
    models = []
    
    # Standard models (.pkl files)
    for pkl_file in model_dir.glob("*.pkl"):
        models.append(str(pkl_file))
    
    # AutoGluon models (directories)
    autogluon_dir = model_dir / config.autogluon.model_subdirectory
    if autogluon_dir.exists():
        for ag_dir in autogluon_dir.iterdir():
            if ag_dir.is_dir():
                models.append(str(ag_dir))
    
    return models

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        gpu_available=check_gpu_availability(),
        models_available=get_available_models()
    )

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training in the background."""
    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Add job to background tasks
    background_tasks.add_task(
        run_training_job, 
        job_id, 
        request.pipeline_type, 
        request.use_gpu
    )
    
    # Store job metadata
    background_jobs[job_id] = {
        "status": "started",
        "pipeline_type": request.pipeline_type,
        "use_gpu": request.use_gpu,
        "started_at": datetime.now().isoformat()
    }
    
    return TrainingResponse(
        job_id=job_id,
        status="started",
        message=f"Training job {job_id} started",
        timestamp=datetime.now().isoformat()
    )

async def run_training_job(job_id: str, pipeline_type: str, use_gpu: bool):
    """Background task for running training jobs."""
    try:
        background_jobs[job_id]["status"] = "running"
        
        if pipeline_type == "standard":
            run_training_pipeline(use_gpu=use_gpu)
        elif pipeline_type == "autogluon":
            run_autogluon_pipeline(use_gpu=use_gpu)
        elif pipeline_type == "tune":
            run_tuning_pipeline(use_gpu=use_gpu)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["error"] = str(e)
        background_jobs[job_id]["failed_at"] = datetime.now().isoformat()
        logger.error(f"Training job {job_id} failed: {e}")

@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    """Get status of a training job."""
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return background_jobs[job_id]

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    file: UploadFile = File(...),
    model_path: str = Query(..., description="Path to the trained model"),
):
    """Make a prediction on a single uploaded file."""
    if not file.filename.endswith('.csv.txt'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a .csv.txt spectral file"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv.txt') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Make prediction
        prediction = predictor.make_prediction(
            input_file=Path(tmp_file_path),
            model_path=Path(model_path)
        )
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return PredictionResponse(
            predicted_magnesium=prediction,
            model_used=model_path,
            timestamp=datetime.now().isoformat(),
            sample_id=file.filename.replace('.csv.txt', '')
        )
        
    except Exception as e:
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_path: str = Query(..., description="Path to the trained model"),
    reference_file: Optional[UploadFile] = File(None)
):
    """Make batch predictions on multiple uploaded files."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    for file in files:
        if not file.filename.endswith('.csv.txt'):
            raise HTTPException(
                status_code=400,
                detail=f"All files must be .csv.txt format. Invalid: {file.filename}"
            )
    
    try:
        # Create temporary directory for batch processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save all uploaded files
            for file in files:
                file_path = temp_path / file.filename
                content = await file.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
            
            # Save reference file if provided
            reference_path = None
            if reference_file:
                reference_path = temp_path / reference_file.filename
                content = await reference_file.read()
                with open(reference_path, 'wb') as f:
                    f.write(content)
            
            # Make batch predictions
            results = predictor.make_batch_predictions(
                input_dir=temp_path,
                model_path=Path(model_path)
            )
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"batch_predictions_{timestamp}.csv"
            output_path = config.reports_dir / output_filename
            results.to_csv(output_path, index=False)
            
            # Calculate metrics if reference provided
            metrics = None
            if reference_path:
                # Calculate metrics (simplified)
                successful = len(results[results['Status'] == 'Success'])
                total = len(results)
                metrics = {
                    "success_rate": successful / total if total > 0 else 0,
                    "total_samples": total,
                    "successful_predictions": successful,
                    "failed_predictions": total - successful
                }
            
            return BatchPredictionResponse(
                job_id=f"batch_{timestamp}",
                total_samples=len(results),
                successful_predictions=len(results[results['Status'] == 'Success']),
                failed_predictions=len(results[results['Status'] != 'Success']),
                output_file=str(output_path),
                metrics=metrics
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List all available trained models."""
    return {"models": get_available_models()}

@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """Download generated files (reports, models, etc.)."""
    if file_type == "report":
        file_path = config.reports_dir / filename
    elif file_type == "model":
        file_path = config.model_dir / filename
    elif file_type == "log":
        file_path = config.log_dir / filename
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )
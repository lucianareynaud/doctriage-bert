"""
FastAPI service for DocTriage-BERT document classification API.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import tempfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import pandas as pd
from uuid import uuid4
import shutil
import uvicorn
from src.ingest_domains import pdf_to_text

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DocTriage-BERT API",
    description="API for classifying documents as reports or regulations",
    version="0.1.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
MODEL = None
TOKENIZER = None
ID2LABEL = {0: "reports", 1: "regulations"}
LABEL2ID = {"reports": 0, "regulations": 1}
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/distil-lora-4bit")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")

# Pydantic models for API
class PredictionResult(BaseModel):
    filename: str
    prediction: str
    confidence: float
    text_preview: str
    metadata: Dict[str, Any]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    job_id: str
    status: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    results: Optional[List[PredictionResult]] = None

# Dictionary to store background job status
JOB_STATUS = {}

def load_model():
    """Load the classification model and tokenizer."""
    global MODEL, TOKENIZER
    
    if MODEL is None or TOKENIZER is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        try:
            # Load tokenizer
            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
            
            # Load model
            MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                MODEL = MODEL.to("cuda")
            
            MODEL.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

def predict_text(text: str) -> dict:
    """
    Make prediction on a text string.
    
    Returns:
        dict: Dictionary with prediction and confidence
    """
    # Ensure model is loaded
    if MODEL is None or TOKENIZER is None:
        load_model()
    
    # Tokenize text
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
    # Get prediction and confidence
    prediction_idx = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, prediction_idx].item()
    prediction = ID2LABEL[prediction_idx]
    
    return {
        "prediction": prediction,
        "confidence": confidence
    }

def process_document_batch(job_id: str, file_paths: List[str]):
    """Process a batch of documents as a background task."""
    try:
        num_files = len(file_paths)
        results = []
        
        for idx, file_path in enumerate(file_paths):
            try:
                # Extract text from PDF
                text = pdf_to_text(file_path)
                
                # Text preview (first 200 chars)
                text_preview = text[:200] + "..." if len(text) > 200 else text
                
                # Get prediction
                pred_result = predict_text(text)
                
                # Extract filename
                filename = os.path.basename(file_path)
                
                # Simple metadata extraction
                metadata = {
                    "filename": filename,
                    "size_bytes": os.path.getsize(file_path),
                    "path": file_path
                }
                
                results.append(PredictionResult(
                    filename=filename,
                    prediction=pred_result["prediction"],
                    confidence=pred_result["confidence"],
                    text_preview=text_preview,
                    metadata=metadata
                ))
                
                # Update progress
                JOB_STATUS[job_id]["progress"] = (idx + 1) / num_files
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Update job status
        JOB_STATUS[job_id]["status"] = "completed"
        JOB_STATUS[job_id]["results"] = results
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["error"] = str(e)

@app.on_event("startup")
async def startup_event():
    """Initialize the model and upload directory on startup."""
    load_model()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory created at {UPLOAD_DIR}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to DocTriage-BERT API"}

@app.post("/predict/text", response_model=PredictionResult)
async def predict_from_text(text: str = Form(...)):
    """Predict document type from text input."""
    try:
        result = predict_text(text)
        return PredictionResult(
            filename="text_input.txt",
            prediction=result["prediction"],
            confidence=result["confidence"],
            text_preview=text[:200] + "..." if len(text) > 200 else text,
            metadata={"source": "text_input"}
        )
    except Exception as e:
        logger.error(f"Error predicting from text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/file", response_model=PredictionResult)
async def predict_from_file(file: UploadFile = File(...)):
    """Predict document type from a single uploaded PDF file."""
    # Save uploaded file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract text
        text = pdf_to_text(temp_file_path)
        
        # Make prediction
        result = predict_text(text)
        
        # Create response
        return PredictionResult(
            filename=file.filename,
            prediction=result["prediction"],
            confidence=result["confidence"],
            text_preview=text[:200] + "..." if len(text) > 200 else text,
            metadata={
                "filename": file.filename,
                "size_bytes": os.path.getsize(temp_file_path)
            }
        )
    
    except Exception as e:
        logger.error(f"Error predicting from file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Process multiple PDF documents in a batch.
    Returns a job ID that can be used to check status.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create job ID
    job_id = str(uuid4())
    
    # Create directory for files
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save files
    file_paths = []
    for file in files:
        file_path = os.path.join(job_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(file_path)
    
    # Initialize job status
    JOB_STATUS[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "results": None
    }
    
    # Start background task
    background_tasks.add_task(process_document_batch, job_id, file_paths)
    
    return BatchPredictionResponse(
        results=[],
        job_id=job_id,
        status="processing"
    )

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a batch prediction job."""
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = JOB_STATUS[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        results=job_data.get("results")
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if MODEL is None or TOKENIZER is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "model_path": MODEL_PATH}

if __name__ == "__main__":
    # Run the API server directly
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True) 
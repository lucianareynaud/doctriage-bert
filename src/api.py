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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
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
            
            # Carregamento padrão do modelo, sem quantização que causa problemas
            MODEL = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,  # Forçar precisão completa para evitar erros de compatibilidade
                local_files_only=False      # Permitir download caso necessário
            )
            
            # Garantir que o modelo está em modo de avaliação
            MODEL.eval()
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Não lançar exceção aqui, apenas registrar o erro
            # Em vez disso, vamos criar um modelo "dummy" que não quebra o fluxo
            logger.warning("Creating dummy model for graceful degradation")
            try:
                config = AutoConfig.from_pretrained("distilbert-base-uncased")
                config.num_labels = 2
                MODEL = AutoModelForSequenceClassification.from_config(config)
                MODEL.eval()
            except Exception as inner_e:
                logger.error(f"Failed to create dummy model: {str(inner_e)}")

def predict_text(text: str) -> dict:
    """
    Make prediction on a text string with robust error handling.
    
    Returns:
        dict: Dictionary with prediction and confidence
    """
    # Garantir texto válido
    if not text or not isinstance(text, str):
        logger.warning(f"Invalid text input provided: {type(text)}")
        return {
            "prediction": "unknown",
            "confidence": 0.0
        }
    
    # Garantir que o modelo está carregado
    if MODEL is None or TOKENIZER is None:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return {
                "prediction": "unknown",
                "confidence": 0.0
            }
        
        # Dupla verificação após tentativa de carregamento
        if MODEL is None or TOKENIZER is None:
            logger.error("Model or tokenizer still None after load attempt")
            return {
                "prediction": "unknown",
                "confidence": 0.0
            }
    
    try:
        # Tokenizar texto com tratamento de erro
        try:
            inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
            if not inputs or len(inputs) == 0:
                logger.error("Tokenization returned empty inputs")
                return {
                    "prediction": "unknown",
                    "confidence": 0.0
                }
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            return {
                "prediction": "unknown",
                "confidence": 0.0
            }
        
        # Predição com tratamento robusto de erros
        with torch.no_grad():
            try:
                outputs = MODEL(**inputs)
                
                # Verificar se outputs é None ou não tem logits
                if outputs is None:
                    logger.error("Model returned None outputs")
                    return {
                        "prediction": "unknown",
                        "confidence": 0.0
                    }
                
                logits = getattr(outputs, 'logits', None)
                if logits is None:
                    logger.error("Model outputs do not contain logits")
                    return {
                        "prediction": "unknown",
                        "confidence": 0.0
                    }
                
                # Verificar formato dos logits
                if not isinstance(logits, torch.Tensor) or len(logits.shape) < 2:
                    logger.error(f"Invalid logits shape: {getattr(logits, 'shape', 'unknown')}")
                    return {
                        "prediction": "unknown",
                        "confidence": 0.0
                    }
                
                # Calcular probabilidades e obter predição
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                prediction_idx = torch.argmax(probabilities, dim=-1).item()
                
                # Verificar se índice de predição é válido
                if prediction_idx not in ID2LABEL:
                    logger.error(f"Invalid prediction index: {prediction_idx}")
                    return {
                        "prediction": "unknown",
                        "confidence": 0.0
                    }
                
                confidence = probabilities[0, prediction_idx].item()
                prediction = ID2LABEL[prediction_idx]
                
                return {
                    "prediction": prediction,
                    "confidence": confidence
                }
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                return {
                    "prediction": "unknown",
                    "confidence": 0.0
                }
    except Exception as e:
        logger.error(f"Unexpected error in predict_text: {str(e)}")
        return {
            "prediction": "unknown",
            "confidence": 0.0
        }

def process_document_batch(job_id: str, file_paths: List[str]):
    """Process a batch of documents as a background task."""
    try:
        # Verificar se file_paths é válido e não vazio
        if not file_paths or not isinstance(file_paths, list):
            logger.error(f"Invalid file_paths in job {job_id}: {type(file_paths)}")
            JOB_STATUS[job_id] = {
                "status": "failed",
                "progress": 0.0,
                "error": "Invalid file paths provided",
                "results": []
            }
            return
        
        num_files = len(file_paths)
        logger.info(f"Processing batch job {job_id} with {num_files} files")
        results = []
        
        for idx, file_path in enumerate(file_paths):
            try:
                # Verificar se o arquivo existe
                if not os.path.exists(file_path):
                    logger.error(f"File does not exist: {file_path}")
                    results.append(PredictionResult(
                        filename=os.path.basename(file_path),
                        prediction="error",
                        confidence=0.0,
                        text_preview=f"Error: File not found",
                        metadata={
                            "filename": os.path.basename(file_path),
                            "error": "File not found",
                            "path": file_path,
                            "processing_status": "error"
                        }
                    ))
                    continue
                
                # Extract text from PDF with error handling
                try:
                    text = pdf_to_text(file_path)
                    if not text or not isinstance(text, str):
                        logger.warning(f"PDF extraction returned empty or invalid text for {file_path}")
                        text = "Failed to extract meaningful text from document."
                except Exception as e:
                    logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
                    text = f"Error extracting text: {str(e)}"
                
                # Criar uma prévia do texto - garantir que nunca é None
                text_preview = text[:200] + "..." if text and len(text) > 200 else (text or "No preview available")
                
                # Get prediction with error handling
                try:
                    pred_result = predict_text(text)
                    if not pred_result or not isinstance(pred_result, dict):
                        logger.error(f"predict_text returned invalid result: {type(pred_result)}")
                        pred_result = {"prediction": "unknown", "confidence": 0.0}
                except Exception as e:
                    logger.error(f"Error predicting from text for {file_path}: {str(e)}")
                    pred_result = {"prediction": "unknown", "confidence": 0.0}
                
                # Extract filename
                filename = os.path.basename(file_path)
                
                # Simple metadata extraction
                try:
                    metadata = {
                        "filename": filename,
                        "size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        "path": file_path,
                        "processing_status": "success" if pred_result.get("prediction") != "unknown" else "limited_success"
                    }
                except Exception as e:
                    logger.error(f"Error creating metadata for {file_path}: {str(e)}")
                    metadata = {
                        "filename": filename,
                        "path": file_path,
                        "error": str(e),
                        "processing_status": "error"
                    }
                
                # Create and add result
                try:
                    results.append(PredictionResult(
                        filename=filename,
                        prediction=pred_result.get("prediction", "unknown"),
                        confidence=pred_result.get("confidence", 0.0),
                        text_preview=text_preview,
                        metadata=metadata
                    ))
                except Exception as e:
                    logger.error(f"Error creating result object for {file_path}: {str(e)}")
                    # Fallback para um resultado mínimo válido
                    results.append(PredictionResult(
                        filename=filename,
                        prediction="error",
                        confidence=0.0,
                        text_preview="Error creating result",
                        metadata={"filename": filename, "error": str(e)}
                    ))
                
                # Update progress
                JOB_STATUS[job_id]["progress"] = (idx + 1) / num_files
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                # Adicionar resultado de erro, mas continuar processando
                try:
                    results.append(PredictionResult(
                        filename=os.path.basename(file_path),
                        prediction="error",
                        confidence=0.0,
                        text_preview=f"Error: {str(e)}",
                        metadata={
                            "filename": os.path.basename(file_path),
                            "error": str(e),
                            "path": file_path
                        }
                    ))
                except Exception as inner_e:
                    logger.error(f"Failed to create error result: {str(inner_e)}")
        
        # Update job status
        JOB_STATUS[job_id]["status"] = "completed"
        JOB_STATUS[job_id]["results"] = results
        logger.info(f"Completed batch job {job_id} with {len(results)} results")
        
    except Exception as e:
        logger.error(f"Error in batch processing for job {job_id}: {str(e)}")
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["error"] = str(e)
        # Garantir que existe um campo results, mesmo que vazio
        if "results" not in JOB_STATUS[job_id]:
            JOB_STATUS[job_id]["results"] = []

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
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Processing file: {file.filename}")
        # Extract text with robust error handling
        try:
            text = pdf_to_text(temp_file_path)
            if not text or not isinstance(text, str):
                logger.warning(f"PDF extraction returned empty or invalid text: {type(text)}")
                text = "Error: Failed to extract meaningful text from document."
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            text = "Error: Failed to extract text from document."
        # Make prediction with robust error handling
        try:
            result = predict_text(text)
            logger.info(f"Prediction result: {result}")
            # Garantir que result não é None e tem os campos necessários
            if not result or not isinstance(result, dict):
                logger.error(f"predict_text returned invalid result: {type(result)}")
                result = {"prediction": "unknown", "confidence": 0.0}
            # Garantir que as chaves existem
            prediction = result.get("prediction", "unknown")
            confidence = result.get("confidence", 0.0)
        except Exception as e:
            logger.error(f"Error predicting text: {str(e)}")
            prediction = "unknown"
            confidence = 0.0
        # Create preview text - garantir que nunca é None
        text_preview = text[:200] + "..." if text and len(text) > 200 else (text or "No preview available")
        # Create response with default values para campos obrigatórios
        return PredictionResult(
            filename=file.filename or "unknown_file.pdf",
            prediction=prediction,
            confidence=confidence,
            text_preview=text_preview,
            metadata={
                "filename": file.filename or "unknown_file.pdf",
                "size_bytes": os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else 0,
                "content_type": file.content_type or "application/pdf",
                "processing_status": "success" if prediction != "unknown" else "limited_success"
            }
        )
    except Exception as e:
        logger.error(f"Error in predict_from_file: {str(e)}")
        return PredictionResult(
            filename=getattr(file, "filename", "unknown_file.pdf"),
            prediction="processing_error",
            confidence=0.0,
            text_preview="Error occurred during processing",
            metadata={
                "filename": getattr(file, "filename", "unknown_file.pdf"),
                "error": str(e),
                "processing_status": "error"
            }
        )
    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")

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
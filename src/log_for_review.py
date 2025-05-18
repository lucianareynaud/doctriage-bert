"""
Log documents to Argilla for human-in-the-loop review
"""

import os
import argparse
import argilla as rg
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Log documents to Argilla for human review"
    )
    parser.add_argument(
        "--model-path", type=str, default="outputs/distil-lora-4bit",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--test-data", type=str, default="data/test",
        help="Path to the test data directory"
    )
    parser.add_argument(
        "--argilla-api-url", type=str, default="http://localhost:6900",
        help="Argilla API URL"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="doctriage_review",
        help="Name of the Argilla dataset to create"
    )
    parser.add_argument(
        "--workspace", type=str, default="admin",
        help="Argilla workspace name"
    )
    parser.add_argument(
        "--api-key", type=str, default="admin.apikey",
        help="Argilla API key"
    )
    parser.add_argument(
        "--max-examples", type=int, default=100,
        help="Maximum number of examples to log"
    )
    parser.add_argument(
        "--min-text-length", type=int, default=50,
        help="Minimum text length to include in review"
    )
    return parser.parse_args()

def init_argilla(args):
    """Initialize Argilla client."""
    rg.init(
        api_url=args.argilla_api_url,
        api_key=args.api_key,
        workspace=args.workspace
    )
    
    # Check if dataset exists, if so, delete it
    try:
        existing_datasets = rg.list_datasets()
        if args.dataset_name in [d.name for d in existing_datasets]:
            logger.info(f"Deleting existing dataset: {args.dataset_name}")
            rg.delete_dataset(args.dataset_name)
    except Exception as e:
        logger.warning(f"Error checking existing datasets: {str(e)}")
    
    # Create new dataset
    dataset = rg.TextClassificationDataset(
        name=args.dataset_name,
        metadata={
            "description": "Document classification review (reports vs. regulations)"
        },
        guidelines="Review the document classifications and correct any misclassifications.",
        labels=["reports", "regulations"]
    )
    
    return dataset

def load_model(model_path):
    """Load the classification model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def predict_text(text, model, tokenizer):
    """Predict document type for a text sample."""
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
    # Get prediction
    prediction_idx = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, prediction_idx].item()
    prediction = "reports" if prediction_idx == 0 else "regulations"
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "reports": probabilities[0, 0].item(),
            "regulations": probabilities[0, 1].item(),
        }
    }

def load_test_data(data_dir):
    """Load test data from Parquet files."""
    test_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith('.parquet')]
    
    if not test_files:
        logger.error(f"No parquet files found in {data_dir}")
        raise ValueError(f"No parquet files found in {data_dir}")
    
    # Load all parquet files into a single dataset
    test_ds = load_dataset("parquet", data_files=test_files)["train"]
    logger.info(f"Loaded {len(test_ds)} examples from {data_dir}")
    
    return test_ds

def create_argilla_records(dataset, model, tokenizer, args):
    """Create Argilla records from dataset."""
    records = []
    
    # Process each example
    for i, example in enumerate(dataset):
        if i >= args.max_examples:
            break
            
        text = example["text"]
        
        # Skip if text is too short
        if len(text) < args.min_text_length:
            continue
            
        true_label = example["domain"]
        
        # Get model prediction
        prediction = predict_text(text, model, tokenizer)
        
        # Create text preview
        text_preview = text[:500] + "..." if len(text) > 500 else text
        
        # Create Argilla record
        record = rg.TextClassificationRecord(
            text=text_preview,
            prediction=prediction["prediction"],
            prediction_agent="model",
            annotation=true_label,  # Pre-fill with true label
            annotation_agent="dataset",
            metadata={
                "file_path": example.get("file_path", ""),
                "filename": example.get("filename", os.path.basename(example.get("file_path", ""))),
                "confidence": prediction["confidence"],
                "reports_prob": prediction["probabilities"]["reports"],
                "regulations_prob": prediction["probabilities"]["regulations"]
            }
        )
        records.append(record)
        
    logger.info(f"Created {len(records)} Argilla records")
    return records

def main():
    args = parse_args()
    
    # Initialize Argilla
    dataset = init_argilla(args)
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Load test data
    test_data = load_test_data(args.test_data)
    
    # Create Argilla records
    records = create_argilla_records(test_data, model, tokenizer, args)
    
    # Log records to Argilla
    if records:
        dataset.add_records(records)
        logger.info(f"Logged {len(records)} records to Argilla dataset: {args.dataset_name}")
        logger.info(f"Visit {args.argilla_api_url} to review the documents")
    else:
        logger.warning("No records were created. Check your data and filters.")

if __name__ == "__main__":
    main() 
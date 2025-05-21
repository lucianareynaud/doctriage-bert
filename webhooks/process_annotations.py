#!/usr/bin/env python3
"""
Process Argilla annotations for model retraining.

This script:
1. Retrieves annotations from Argilla
2. Prepares a dataset for retraining
3. Triggers the retraining process
"""

import os
import sys
import json
import argparse
import requests
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Argilla configuration
ARGILLA_API_URL = "http://localhost:6900"
ARGILLA_USERNAME = "admin"  # Default admin user in Argilla
ARGILLA_PASSWORD = "12345678"  # From docker-compose.yml
ARGILLA_API_KEY = "argilla.apikey"  # From docker-compose.yml
WORKSPACE = "admin"  # Default workspace
DATASET_NAME = "doctriage_classification"

def parse_args():
    parser = argparse.ArgumentParser(description="Process Argilla annotations for model retraining")
    parser.add_argument("--annotation-file", type=str, help="Path to write the annotation file")
    parser.add_argument("--min-annotations", type=int, default=5,
                      help="Minimum number of annotations required to trigger retraining")
    parser.add_argument("--model-path", type=str, default="outputs/distil-lora-4bit",
                      help="Path to the current model")
    return parser.parse_args()

def get_annotations():
    """Retrieve annotations from Argilla."""
    try:
        # Set up authentication
        auth = (ARGILLA_USERNAME, ARGILLA_PASSWORD)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Add API key to headers
        api_headers = headers.copy()
        api_headers["X-API-KEY"] = ARGILLA_API_KEY
        
        # Query annotated records
        url = f"{ARGILLA_API_URL}/api/v1/datasets/{DATASET_NAME}/TextClassification:records"
        params = {"status": "Submitted"}  # Get submitted annotations
        
        # Try API key first
        response = requests.get(url, params=params, headers=api_headers)
        
        if response.status_code != 200:
            # Fall back to basic auth
            logger.info("API key authentication failed, trying Basic Auth...")
            response = requests.get(url, params=params, auth=auth, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to retrieve annotations: {response.status_code}")
            logger.error(response.text)
            return []
        
        # Parse and process the records
        result = response.json()
        records = result.get("items", [])
        
        # Filter for records with annotations
        annotated_records = [record for record in records if record.get("annotation") is not None]
        logger.info(f"Found {len(annotated_records)} annotated records")
        
        return annotated_records
    except Exception as e:
        logger.error(f"Error retrieving annotations: {e}")
        return []

def prepare_dataset(annotations, output_file):
    """Prepare dataset for retraining."""
    try:
        # Structure the annotations for training
        training_data = []
        for record in annotations:
            # Extract the annotation label
            annotation = record.get("annotation")
            annotation_label = annotation[0][0] if annotation and len(annotation) > 0 else None
            
            if not annotation_label:
                continue
                
            # Create training example
            example = {
                "text": record.get("text", ""),
                "label": annotation_label,
                "metadata": record.get("metadata", {})
            }
            training_data.append(example)
        
        # Write to the output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)
            
        logger.info(f"Prepared {len(training_data)} examples for training")
        return len(training_data)
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return 0

def trigger_retraining(annotation_file, model_path):
    """Trigger the model retraining process."""
    try:
        # Generate a timestamp for the new model
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"outputs/distil-lora-retrained-{timestamp}"
        
        # Create placeholder for future trained model
        os.makedirs(output_dir, exist_ok=True)
        
        # Write model information
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump({
                "base_model": model_path,
                "annotation_file": annotation_file,
                "timestamp": timestamp,
                "annotation_count": sum(1 for line in open(annotation_file)) if os.path.exists(annotation_file) else 0,
                "status": "ready_for_training"
            }, f, indent=2)
        
        logger.info(f"✅ Prepared for retraining. When ML dependencies are installed, run:")
        logger.info(f"python src/retrain.py --base-model {model_path} --annotation-file {annotation_file} --output-dir {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error preparing for retraining: {e}")
        return False

def main():
    args = parse_args()
    
    # Generate default annotation file if not provided
    if not args.annotation_file:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.annotation_file = f"data/annotations/annotations-{timestamp}.json"
    
    # Get annotations from Argilla
    annotations = get_annotations()
    
    # Check if we have enough annotations to retrain
    if len(annotations) < args.min_annotations:
        logger.info(f"Not enough annotations ({len(annotations)}/{args.min_annotations}) to trigger retraining")
        return
    
    # Prepare dataset
    count = prepare_dataset(annotations, args.annotation_file)
    if count < args.min_annotations:
        logger.info(f"Not enough valid examples ({count}/{args.min_annotations}) to trigger retraining")
        return
    
    # Prepare for retraining
    success = trigger_retraining(args.annotation_file, args.model_path)
    
    if success:
        print(f"✅ Successfully processed {count} annotations and prepared for retraining!")
        print(f"✅ Annotation file saved to: {args.annotation_file}")
    else:
        print(f"❌ Failed to process annotations. Check the logs for details.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Argilla setup script for DocTriage-BERT

This script:
1. Initializes Argilla with appropriate credentials
2. Creates a dataset with the right configuration for annotation
3. Adds sample documents for review
4. Sets up the pipeline for annotation and retraining

Used by start.sh as part of the automated setup process.
"""

import os
import sys
import logging
import requests
import json
import time
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
ARGILLA_API_URL = "http://localhost:6900"
ARGILLA_USERNAME = "admin"  # Default admin user in Argilla
ARGILLA_PASSWORD = "12345678"  # From docker-compose.yml
ARGILLA_API_KEY = "argilla.apikey"  # From docker-compose.yml
WORKSPACE = "admin"  # Default workspace
DATASET_NAME = "doctriage_classification"

# Sample documents for annotation
SAMPLE_DOCUMENTS = [
    {
        "text": "This quarterly report contains financial information and operational metrics for Q1 2025. The company has exceeded revenue expectations by 15%, with a total revenue of $2.3 million for the quarter. The board has approved a dividend of $0.25 per share.",
        "prediction": "reports",
        "confidence": 0.92,
        "reports_prob": 0.92,
        "regulations_prob": 0.08,
        "filename": "sample_report_1.pdf"
    },
    {
        "text": "All organizations must comply with the new data privacy regulations effective June 1, 2025. The regulation requires explicit consent for data collection and imposes fines of up to $10 million or 4% of global revenue for violations. Organizations have 90 days to update their privacy policies.",
        "prediction": "regulations",
        "confidence": 0.88,
        "reports_prob": 0.12,
        "regulations_prob": 0.88,
        "filename": "sample_regulation_1.pdf"
    },
    {
        "text": "The company's annual financial report shows a 12% increase in revenue and a 5% decrease in operating costs. Net profit increased by 18% to $5.2 million. The board of directors recommends a dividend of $0.50 per share.",
        "prediction": "reports",
        "confidence": 0.95,
        "reports_prob": 0.95,
        "regulations_prob": 0.05,
        "filename": "sample_report_2.pdf"
    },
    {
        "text": "New environmental regulations require all manufacturing facilities to reduce carbon emissions by 30% by 2030. Companies must submit annual progress reports starting January 2026. Non-compliance will result in fines of $500 per ton of excess emissions.",
        "prediction": "regulations",
        "confidence": 0.91,
        "reports_prob": 0.09,
        "regulations_prob": 0.91,
        "filename": "sample_regulation_2.pdf"
    }
]

def check_argilla_running():
    """Check if Argilla server is running."""
    try:
        response = requests.get(f"{ARGILLA_API_URL}/api/v1/status", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Argilla server is running")
            return True
        else:
            logger.error(f"❌ Argilla server returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Argilla server: {e}")
        return False

def create_argilla_dataset():
    """Create a new dataset in Argilla using the REST API."""
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
        
        # Check if dataset exists
        response = requests.get(
            f"{ARGILLA_API_URL}/api/v1/datasets", 
            headers=api_headers
        )
        
        if response.status_code != 200:
            # Fall back to basic auth if API key didn't work
            logger.info("API key authentication failed, trying Basic Auth...")
            response = requests.get(
                f"{ARGILLA_API_URL}/api/v1/datasets", 
                auth=auth,
                headers=headers
            )
        
        if response.status_code == 200:
            datasets = response.json()
            for dataset in datasets.get("items", []):
                if dataset.get("name") == DATASET_NAME:
                    logger.info(f"Dataset '{DATASET_NAME}' already exists, deleting it")
                    # Delete existing dataset
                    delete_response = requests.delete(
                        f"{ARGILLA_API_URL}/api/v1/datasets/{DATASET_NAME}",
                        headers=api_headers
                    )
                    
                    if delete_response.status_code != 200:
                        # Fall back to basic auth
                        delete_response = requests.delete(
                            f"{ARGILLA_API_URL}/api/v1/datasets/{DATASET_NAME}",
                            auth=auth,
                            headers=headers
                        )
                    
                    if delete_response.status_code == 200:
                        logger.info(f"✅ Deleted existing dataset: {DATASET_NAME}")
                    else:
                        logger.warning(f"⚠️ Failed to delete dataset: {delete_response.status_code}")
                        logger.warning(delete_response.text)
                    # Small delay to ensure deletion is processed
                    time.sleep(2)
                    break

        # Create new dataset
        payload = {
            "name": DATASET_NAME,
            "task": "TextClassification",
            "metadata": {
                "description": "Document classification review (reports vs. regulations)"
            },
            "guidelines": "Review document classifications and correct any errors. This feedback will be used to improve model quality through retraining.",
            "allow_extra_metadata": True,
            "labels": ["reports", "regulations"]
        }
        
        # Try with API key first
        response = requests.post(
            f"{ARGILLA_API_URL}/api/v1/datasets",
            json=payload,
            headers=api_headers
        )
        
        if response.status_code != 201 and response.status_code != 200:
            # Fall back to basic auth
            logger.info("API key authentication failed for dataset creation, trying Basic Auth...")
            response = requests.post(
                f"{ARGILLA_API_URL}/api/v1/datasets",
                json=payload,
                auth=auth,
                headers=headers
            )
        
        if response.status_code == 201 or response.status_code == 200:
            logger.info(f"✅ Created dataset: {DATASET_NAME}")
            return True
        else:
            logger.error(f"❌ Failed to create dataset: {response.status_code}")
            logger.error(response.text)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating dataset: {e}")
        return False

def add_sample_documents():
    """Add sample documents to the dataset."""
    success_count = 0
    
    # Set up authentication
    auth = (ARGILLA_USERNAME, ARGILLA_PASSWORD)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Add API key to headers
    api_headers = headers.copy()
    api_headers["X-API-KEY"] = ARGILLA_API_KEY
    
    for doc in SAMPLE_DOCUMENTS:
        try:
            # Prepare record for Argilla API
            record = {
                "text": doc["text"],
                "prediction": [[doc["prediction"], doc["confidence"]]],
                "metadata": {
                    "filename": doc["filename"],
                    "confidence": doc["confidence"],
                    "reports_prob": doc["reports_prob"],
                    "regulations_prob": doc["regulations_prob"],
                    "source": "sample"
                }
            }
            
            # Try API key first
            response = requests.post(
                f"{ARGILLA_API_URL}/api/v1/datasets/{DATASET_NAME}/TextClassification:records",
                json=record,
                headers=api_headers
            )
            
            if response.status_code != 201 and response.status_code != 200:
                # Fall back to basic auth
                response = requests.post(
                    f"{ARGILLA_API_URL}/api/v1/datasets/{DATASET_NAME}/TextClassification:records",
                    json=record,
                    auth=auth,
                    headers=headers
                )
            
            if response.status_code == 201 or response.status_code == 200:
                success_count += 1
            else:
                logger.warning(f"⚠️ Failed to add record: {response.status_code}")
                logger.warning(response.text)
                
        except Exception as e:
            logger.warning(f"⚠️ Error adding record: {e}")
    
    logger.info(f"✅ Added {success_count}/{len(SAMPLE_DOCUMENTS)} sample documents to Argilla")
    return success_count > 0

def create_webhook_script():
    """Create a script to process annotation webhooks for retraining."""
    try:
        # Create the webhooks directory if it doesn't exist
        Path("webhooks").mkdir(exist_ok=True)
        
        # Create the webhook processing script
        webhook_script_path = "webhooks/process_annotations.py"
        with open(webhook_script_path, "w") as f:
            f.write('''#!/usr/bin/env python3
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
ARGILLA_API_KEY = "admin.apikey"
WORKSPACE = "admin"
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
        # Query annotated records
        url = f"{ARGILLA_API_URL}/api/v1/datasets/{DATASET_NAME}/TextClassification:records"
        params = {"status": "Submitted"}  # Get submitted annotations
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to retrieve annotations: {response.status_code}")
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
        
        # Run the retraining command
        cmd = [
            "python", "src/retrain.py",
            "--base-model", model_path,
            "--annotation-file", annotation_file,
            "--output-dir", output_dir,
            "--epochs", "2",
            "--batch-size", "4"
        ]
        
        logger.info(f"Triggering retraining with command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            logger.info(f"Retraining successful! New model saved to {output_dir}")
            logger.info("To use the new model, update MODEL_PATH in docker-compose.yml")
            return True
        else:
            logger.error(f"Retraining failed: {process.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        return False

def main():
    args = parse_args()
    
    # Generate default annotation file if not provided
    if not args.annotation_file:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.annotation_file = f"data/annotations-{timestamp}.json"
    
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
    
    # Trigger retraining
    success = trigger_retraining(args.annotation_file, args.model_path)
    
    if success:
        print(f"✅ Model retrained successfully with {count} annotations!")
    else:
        print(f"❌ Model retraining failed. Check the logs for details.")

if __name__ == "__main__":
    main()
''')

        # Make the script executable
        os.chmod(webhook_script_path, 0o755)
        
        logger.info(f"✅ Created annotation processing script: {webhook_script_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating webhook script: {e}")
        return False

def create_retrain_script():
    """Create a script for retraining the model with annotations."""
    try:
        retrain_script_path = "src/retrain.py"
        
        # Only create if it doesn't exist
        if os.path.exists(retrain_script_path):
            logger.info(f"✅ Retrain script already exists: {retrain_script_path}")
            return True
            
        with open(retrain_script_path, "w") as f:
            f.write('''#!/usr/bin/env python3
"""
Retrain the model with annotations from Argilla.

This script:
1. Takes the base model and annotations
2. Fine-tunes the model with the new annotated data
3. Saves the retrained model
"""

import os
import sys
import json
import argparse
import logging
import torch
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Retrain the model with annotations")
    parser.add_argument("--base-model", type=str, required=True,
                      help="Path to the base model to retrain")
    parser.add_argument("--annotation-file", type=str, required=True,
                      help="Path to the annotation file")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Path to save the retrained model")
    parser.add_argument("--epochs", type=int, default=2,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                      help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                      help="Learning rate")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                      help="Gradient accumulation steps")
    return parser.parse_args()

def load_annotations(file_path):
    """Load annotations from JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Create a dataset
        dataset_dict = {
            "text": [item["text"] for item in data],
            "label": [item["label"] for item in data]
        }
        
        # Convert labels to IDs
        label_map = {"reports": 0, "regulations": 1}
        dataset_dict["label"] = [label_map.get(label, 0) for label in dataset_dict["label"]]
        
        # Create HF dataset
        dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Loaded {len(dataset)} annotations from {file_path}")
        
        return dataset
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        sys.exit(1)

def tokenize_data(dataset, tokenizer):
    """Tokenize the dataset."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def retrain_model(args):
    """Retrain the model with annotations."""
    try:
        # Load the base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
        
        # Set up LoRA configuration (if not already using PEFT)
        if not isinstance(model, PeftModel):
            logger.info("Setting up LoRA for fine-tuning")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,  # Rank
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                target_modules=["q_lin", "v_lin"]  # Adjust based on model architecture
            )
            model = get_peft_model(model, lora_config)
        
        # Load and preprocess annotations
        dataset = load_annotations(args.annotation_file)
        
        # Split dataset into train/validation
        dataset = dataset.train_test_split(test_size=0.2)
        
        # Tokenize the datasets
        tokenized_train = tokenize_data(dataset["train"], tokenizer)
        tokenized_valid = tokenize_data(dataset["test"], tokenizer)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            learning_rate=args.lr,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,  # Set to True for GPU training
            logging_dir=f"{args.output_dir}/logs",
            logging_steps=10,
            report_to="none"
        )
        
        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Starting model retraining...")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving retrained model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        return True
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        return False

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Retrain the model
    success = retrain_model(args)
    
    if success:
        logger.info(f"✅ Model successfully retrained and saved to {args.output_dir}")
        print(f"✅ Model successfully retrained and saved to {args.output_dir}")
    else:
        logger.error("❌ Model retraining failed")
        print("❌ Model retraining failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
''')

        # Make the script executable
        os.chmod(retrain_script_path, 0o755)
        
        logger.info(f"✅ Created retraining script: {retrain_script_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating retrain script: {e}")
        return False

def create_cron_job():
    """Create a cron job script to check for new annotations."""
    try:
        cron_script_path = "webhooks/check_annotations.sh"
        
        # Create the webhooks directory if it doesn't exist
        Path("webhooks").mkdir(exist_ok=True)
        
        with open(cron_script_path, "w") as f:
            f.write('''#!/bin/bash

# Script to check for annotations and trigger retraining
# Add to crontab to run periodically:
# 0 * * * * /path/to/check_annotations.sh >> /path/to/annotation_checks.log 2>&1

# Set the working directory to the project root
cd "$(dirname "$0")/.."

# Run the annotation processor
python webhooks/process_annotations.py --min-annotations 5

# Check exit status
if [ $? -eq 0 ]; then
  echo "$(date): Successfully processed annotations"
else
  echo "$(date): Failed to process annotations"
fi
''')
        
        # Make the script executable
        os.chmod(cron_script_path, 0o755)
        
        logger.info(f"✅ Created cron job script: {cron_script_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating cron job script: {e}")
        return False

def main():
    print("Setting up Argilla and annotation pipeline for DocTriage-BERT...")
    
    # Check if Argilla is running
    if not check_argilla_running():
        print("❌ Argilla server is not running. Please start it with docker-compose up -d")
        sys.exit(1)
    
    # Create dataset
    if not create_argilla_dataset():
        print("❌ Failed to create Argilla dataset")
        sys.exit(1)
    
    # Add sample documents
    if add_sample_documents():
        print("✅ Successfully added sample documents to Argilla")
    else:
        print("⚠️ Failed to add sample documents, but dataset was created")
    
    # Create annotation processing script
    if create_webhook_script():
        print("✅ Created annotation processing pipeline")
    else:
        print("⚠️ Failed to create annotation processing pipeline")
    
    # Create retraining script
    if create_retrain_script():
        print("✅ Created model retraining script")
    else:
        print("⚠️ Failed to create model retraining script")
    
    # Create cron job script
    if create_cron_job():
        print("✅ Created cron job for annotation checking")
    else:
        print("⚠️ Failed to create cron job script")
    
    print("\n" + "="*60)
    print("✅ ARGILLA AND ANNOTATION PIPELINE SETUP COMPLETE!")
    print("="*60)
    print(f"🔗 Access Argilla at: {ARGILLA_API_URL}")
    print("🔑 Login credentials:")
    print("    Username: argilla")
    print("    Password: 12345678")
    print("\n📝 Annotation and Retraining Pipeline:")
    print("  1. Annotate documents in Argilla")
    print("  2. Process annotations: python webhooks/process_annotations.py")
    print("  3. View retrained models in outputs/ directory")
    print("  4. Set up scheduled retraining by adding to crontab:")
    print("     0 * * * * cd /path/to/doctriage-bert && webhooks/check_annotations.sh")
    
    # Success
    sys.exit(0)

if __name__ == "__main__":
    main() 
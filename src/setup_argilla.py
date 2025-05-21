#!/usr/bin/env python3
"""
Setup Argilla for DocTriage-BERT project

This script:
1. Checks if Argilla is running, starts it if needed
2. Creates the dataset with proper configuration
3. Logs sample documents for review
4. Opens the browser with auto-login
"""

import os
import sys
import argparse
import time
import webbrowser
import subprocess
from pathlib import Path
import logging
import argilla as rg
from log_for_review import load_model, load_test_data, create_argilla_records

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Setup Argilla for DocTriage-BERT project"
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
    return parser.parse_args()

def ensure_argilla_running():
    """Check if Argilla is running, start it if not."""
    try:
        # Check if port 6900 is responding
        import requests
        try:
            response = requests.get("http://localhost:6900", timeout=3)
            if response.status_code == 200:
                logger.info("✅ Argilla is already running")
                return True
        except requests.exceptions.RequestException:
            pass
            
        # Check if Docker containers are running
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=doctriage-argilla"], 
            capture_output=True, 
            text=True
        )
        
        if result.stdout.strip():
            logger.info("✅ Argilla container is running")
            return True
        
        # Start containers if not running
        logger.info("🔄 Argilla is not running. Starting docker-compose...")
        subprocess.run(["docker-compose", "up", "-d"])
        
        # Wait for Argilla to be ready
        logger.info("🔄 Waiting for Argilla to be ready...")
        for i in range(15):
            try:
                response = requests.get("http://localhost:6900", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Argilla is now running")
                    return True
            except requests.exceptions.RequestException:
                pass
                
            print(".", end="", flush=True)
            time.sleep(2)
        
        logger.error("❌ Timed out waiting for Argilla to start")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking Argilla status: {str(e)}")
        return False

def init_argilla(args):
    """Initialize Argilla client and dataset."""
    try:
        logger.info(f"🔄 Initializing Argilla client with URL: {args.argilla_api_url}")
        rg.init(
            api_url=args.argilla_api_url,
            api_key=args.api_key,
            workspace=args.workspace
        )
        
        # Check if dataset exists, if so, delete it
        try:
            existing_datasets = rg.list_datasets()
            if args.dataset_name in [d.name for d in existing_datasets]:
                logger.info(f"🔄 Deleting existing dataset: {args.dataset_name}")
                rg.delete_dataset(args.dataset_name)
        except Exception as e:
            logger.warning(f"⚠️ Error checking existing datasets: {str(e)}")
        
        # Create new dataset
        dataset = rg.TextClassificationDataset(
            name=args.dataset_name,
            metadata={
                "description": "Document classification review (reports vs. regulations)"
            },
            guidelines="Review the document classifications and correct any misclassifications.",
            labels=["reports", "regulations"]
        )
        
        logger.info(f"✅ Created Argilla dataset: {args.dataset_name}")
        return dataset
    except Exception as e:
        logger.error(f"❌ Error initializing Argilla: {str(e)}")
        return None

def open_argilla(argilla_url):
    """Open Argilla in the default web browser with auto-login."""
    html_path = Path("argilla_login.html").absolute()
    
    if html_path.exists():
        logger.info(f"🔄 Opening Argilla via {html_path}")
        # Use the system's default web browser to open the HTML file
        webbrowser.open(f"file://{html_path}")
    else:
        logger.info(f"🔄 Opening Argilla directly: {argilla_url}")
        webbrowser.open(argilla_url)
        
    print("\n" + "-" * 60)
    print("Argilla Login Credentials:")
    print("  Username: argilla")
    print("  Password: 12345678")
    print("-" * 60)

def main():
    args = parse_args()
    
    print("\n" + "="*50)
    print("🚀 DOCTRIAGE-BERT ARGILLA SETUP")
    print("="*50 + "\n")
    
    # Ensure Argilla is running
    if not ensure_argilla_running():
        logger.error("❌ Failed to start Argilla. Please check Docker and try again.")
        return
    
    # Initialize Argilla
    dataset = init_argilla(args)
    if not dataset:
        logger.error("❌ Failed to initialize Argilla dataset. Check your connection settings.")
        return
    
    # Load model and test data if available
    try:
        # Look for test data and model
        if os.path.exists(args.test_data) and os.path.exists(args.model_path):
            logger.info(f"🔄 Loading model from {args.model_path}")
            model, tokenizer = load_model(args.model_path)
            
            logger.info(f"🔄 Loading test data from {args.test_data}")
            test_data = load_test_data(args.test_data)
            
            logger.info(f"🔄 Creating Argilla records")
            records = create_argilla_records(test_data, model, tokenizer, args)
            
            # Log records to Argilla
            if records:
                dataset.add_records(records)
                logger.info(f"✅ Logged {len(records)} records to Argilla dataset: {args.dataset_name}")
            else:
                logger.warning("⚠️ No records were created. Check your test data.")
        else:
            # Just set up Argilla without data
            logger.warning("⚠️ Model or test data not found. Setting up Argilla without sample data.")
            logger.info("💡 You can add data later with: python src/log_for_review.py")
    except Exception as e:
        logger.error(f"❌ Error processing test data: {str(e)}")
    
    # Open Argilla in browser
    open_argilla(args.argilla_api_url)
    
    print("\n" + "="*50)
    print("✅ ARGILLA SETUP COMPLETE")
    print("="*50)
    print("\nYou can now annotate documents at http://localhost:6900")

if __name__ == "__main__":
    main() 
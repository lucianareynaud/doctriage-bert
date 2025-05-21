"""
Test Argilla integration by sending a few sample documents for review.
"""

import argilla as rg
import logging
import sys
import os
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_argilla_connection(api_url, api_key, workspace):
    """Test connection to Argilla server."""
    try:
        rg.init(
            api_url=api_url,
            api_key=api_key,
            workspace=workspace
        )
        
        # Try to list datasets - this will fail if connection is not working
        datasets = rg.list_datasets()
        logger.info(f"Successfully connected to Argilla. Found {len(datasets)} datasets.")
        
        for ds in datasets:
            logger.info(f"Dataset: {ds.name}")
            
        return True
    except Exception as e:
        logger.error(f"Error connecting to Argilla: {str(e)}")
        return False

def create_test_dataset(dataset_name):
    """Create a test dataset with a few records."""
    try:
        # Check if dataset exists, if so, delete it
        existing_datasets = rg.list_datasets()
        if dataset_name in [d.name for d in existing_datasets]:
            logger.info(f"Deleting existing dataset: {dataset_name}")
            rg.delete_dataset(dataset_name)
        
        # Create new dataset
        dataset = rg.TextClassificationDataset(
            name=dataset_name,
            metadata={
                "description": "Test dataset for DocTriage-BERT"
            },
            guidelines="Review the document classifications and correct any misclassifications.",
            labels=["reports", "regulations"]
        )
        
        # Create a few sample records
        records = [
            rg.TextClassificationRecord(
                text="This report contains the quarterly financial results for the company.",
                prediction="reports",
                prediction_agent="test",
                metadata={
                    "confidence": 0.95,
                    "source": "test"
                }
            ),
            rg.TextClassificationRecord(
                text="All organizations must comply with the following regulations.",
                prediction="regulations",
                prediction_agent="test",
                metadata={
                    "confidence": 0.92,
                    "source": "test"
                }
            ),
            rg.TextClassificationRecord(
                text="The committee has reviewed the findings and presents this summary report.",
                prediction="reports",
                prediction_agent="test",
                metadata={
                    "confidence": 0.89,
                    "source": "test"
                }
            )
        ]
        
        # Log records to Argilla
        dataset.add_records(records)
        logger.info(f"Created test dataset '{dataset_name}' with {len(records)} records")
        
        return True
    except Exception as e:
        logger.error(f"Error creating test dataset: {str(e)}")
        return False

def main():
    # Argilla connection parameters
    api_url = "http://localhost:6900"
    
    # We have several options for API keys - try each one
    api_keys = [
        "admin.apikey",  # Admin user
        "owner.apikey",  # Owner user
        "6uoAuW032iyvm7WkLSZNcR56_fHneTDWZ2J2Ixi7CQouyLEAFhpP5yK6Ac4Pv_I1zcVqP6qAhqtE6jWGnRKZHpAFQlEE8jhnC25r-IEw8qQ"  # Argilla user
    ]
    workspace = "admin"
    dataset_name = "doctriage_test"
    
    success = False
    for api_key in api_keys:
        logger.info(f"Testing connection with API key: {api_key[:10]}...")
        if test_argilla_connection(api_url, api_key, workspace):
            success = True
            logger.info(f"Connection successful with API key: {api_key[:10]}...")
            
            # Create test dataset
            if create_test_dataset(dataset_name):
                logger.info(f"Test dataset created successfully. Open {api_url} to view it.")
            
            break
    
    if not success:
        logger.error("Failed to connect to Argilla with any API key.")
        sys.exit(1)
    
    logger.info("Argilla integration test completed successfully!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Argilla Configuration Example for DocTriage-BERT

This script shows how to configure Argilla with the latest SDK format
for use with our document classification project.

Run this script after setting up the environment to initialize Argilla
with the correct dataset configuration.
"""

import argilla as rg
import os
from pathlib import Path

# Configuration
ARGILLA_API_URL = "http://localhost:6900"
ARGILLA_API_KEY = "admin.apikey"  # Default key for the Docker setup
WORKSPACE = "admin"  # Default workspace
DATASET_NAME = "doctriage_classification"

def main():
    print("Configuring Argilla for DocTriage-BERT project...")
    
    # Initialize Argilla connection
    rg.init(
        api_url=ARGILLA_API_URL,
        api_key=ARGILLA_API_KEY,
    )
    
    # Check if the dataset already exists
    try:
        datasets = rg.list_datasets()
        dataset_exists = any(ds.name == DATASET_NAME for ds in datasets)
        
        if dataset_exists:
            print(f"Dataset '{DATASET_NAME}' already exists.")
            # You could delete and recreate it here if needed
            return
    except Exception as e:
        print(f"Error checking datasets: {e}")
    
    # Create a feedback dataset for text classification
    dataset = rg.FeedbackDataset.for_text_classification(
        labels=["reports", "regulations"],
        multi_label=False,
        use_markdown=True,
        guidelines="Review document classifications and correct any errors.",
        metadata_properties=[
            rg.MetadataProperty(name="filename", title="Filename"),
            rg.MetadataProperty(name="confidence", title="Model Confidence"),
            rg.MetadataProperty(name="reports_prob", title="Reports Probability"),
            rg.MetadataProperty(name="regulations_prob", title="Regulations Probability"),
        ]
    )
    
    # Push dataset to Argilla
    dataset.push_to_argilla(name=DATASET_NAME, workspace=WORKSPACE)
    print(f"✅ Created dataset '{DATASET_NAME}' in workspace '{WORKSPACE}'")
    
    # Example of how to add records (can be adapted to use real data)
    # This is a simplified example showing the format for the newer Argilla version
    print("ℹ️ To add records, use code like this:")
    print("""
    records = [
        rg.FeedbackRecord(
            fields={
                "text": "This document outlines reporting requirements...",
            },
            metadata={
                "filename": "example1.pdf",
                "confidence": 0.92,
                "reports_prob": 0.92,
                "regulations_prob": 0.08
            },
            status="pending"
        ),
        # More records...
    ]
    dataset.add_records(records)
    """)
    
    print("\n✅ Argilla configuration complete!")
    print(f"🔗 You can access Argilla at: {ARGILLA_API_URL}")
    print("🔑 Login with username: argilla, password: 12345678")

if __name__ == "__main__":
    main() 
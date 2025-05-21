#!/usr/bin/env python3
"""
Simple Argilla setup script for DocTriage-BERT

This script creates a sample dataset directly with the Argilla SDK 
with simplified authentication that's known to work.
"""

import sys
import os
import json
import time
import argparse
import requests
from datetime import datetime

# Sample documents for annotation
SAMPLE_DOCUMENTS = [
    {
        "text": "This quarterly report contains financial information and operational metrics for Q1 2025. The company has exceeded revenue expectations by 15%, with a total revenue of $2.3 million for the quarter. The board has approved a dividend of $0.25 per share.",
        "label": "reports",
        "confidence": 0.92
    },
    {
        "text": "All organizations must comply with the new data privacy regulations effective June 1, 2025. The regulation requires explicit consent for data collection and imposes fines of up to $10 million or 4% of global revenue for violations. Organizations have 90 days to update their privacy policies.",
        "label": "regulations",
        "confidence": 0.88
    },
    {
        "text": "The company's annual financial report shows a 12% increase in revenue and a 5% decrease in operating costs. Net profit increased by 18% to $5.2 million. The board of directors recommends a dividend of $0.50 per share.",
        "label": "reports",
        "confidence": 0.95
    },
    {
        "text": "New environmental regulations require all manufacturing facilities to reduce carbon emissions by 30% by 2030. Companies must submit annual progress reports starting January 2026. Non-compliance will result in fines of $500 per ton of excess emissions.",
        "label": "regulations",
        "confidence": 0.91
    }
]

def parse_args():
    parser = argparse.ArgumentParser(description="Set up Argilla with sample data")
    parser.add_argument("--url", default="http://localhost:6900", help="Argilla server URL")
    parser.add_argument("--dataset", default="doctriage", help="Dataset name")
    
    # Include all potential credentials as options
    parser.add_argument("--username", default=None, help="Argilla username")
    parser.add_argument("--password", default=None, help="Argilla password")
    parser.add_argument("--api-key", default=None, help="Argilla API key")
    
    return parser.parse_args()

def create_argilla_dataset(args):
    """Create Argilla dataset using raw HTTP requests (minimal dependencies)"""
    
    url = args.url.rstrip("/")  # Remove trailing slash if present
    dataset_name = args.dataset
    api_version = "v1"  # Try v1 endpoint first
    
    # Authentication options to try
    auth_options = []
    
    # Add custom auth if provided
    if args.username and args.password:
        auth_options.append({
            "type": "basic", 
            "auth": (args.username, args.password),
            "note": f"Custom ({args.username}:{args.password})"
        })
    
    if args.api_key:
        auth_options.append({
            "type": "api_key", 
            "key": args.api_key,
            "note": f"Custom API key ({args.api_key})"
        })
    
    # Add default options
    auth_options.extend([
        {"type": "basic", "auth": ("admin", "12345678"), "note": "Default admin"},
        {"type": "basic", "auth": ("argilla", "12345678"), "note": "Default annotator"},
        {"type": "basic", "auth": ("admin", "1234"), "note": "Container default admin"},
        {"type": "basic", "auth": ("argilla", "1234"), "note": "Container default annotator"},
        {"type": "api_key", "key": "argilla.apikey", "note": "Default API key"},
    ])
    
    print(f"Checking if Argilla server is running at {url}...")
    
    # Test if server is running
    for version in ["v1", ""]:
        try:
            status_url = f"{url}/api/{version}/status" if version else f"{url}/api/status"
            response = requests.get(status_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Argilla server is running (API {version or 'v0'})")
                api_version = version
                break
        except Exception:
            pass
    else:
        print("❌ Could not connect to Argilla server. Make sure it's running.")
        return False, None
    
    # Try each authentication method
    for auth_option in auth_options:
        auth_type = auth_option["type"]
        note = auth_option.get("note", "")
        print(f"\nTrying authentication: {note}")
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if auth_type == "api_key":
            headers["X-API-KEY"] = auth_option["key"]
            auth = None
        else:
            auth = auth_option["auth"]
        
        # Test authentication
        try:
            # Different endpoints for v1 vs v0
            if api_version == "v1":
                endpoint = "me" if auth_type == "api_key" else "datasets"
                auth_url = f"{url}/api/v1/{endpoint}"
            else:
                auth_url = f"{url}/api/datasets" 
            
            response = requests.get(auth_url, headers=headers, auth=auth, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Authentication successful!")
                
                # Try to create dataset
                if api_version == "v1":
                    dataset_url = f"{url}/api/v1/datasets"
                    payload = {
                        "name": dataset_name,
                        "task": "TextClassification",
                        "metadata": {
                            "description": "Document classification review (reports vs. regulations)"
                        },
                        "guidelines": "Review document classifications and correct any errors.",
                        "allow_extra_metadata": True,
                        "labels": ["reports", "regulations"]
                    }
                else: 
                    dataset_url = f"{url}/api/datasets"
                    payload = {
                        "name": dataset_name,
                        "task": "TextClassification"
                    }
                
                # Check if dataset exists, delete if it does
                try:
                    # List datasets
                    if api_version == "v1":
                        datasets_url = f"{url}/api/v1/datasets"
                    else:
                        datasets_url = f"{url}/api/datasets"
                        
                    response = requests.get(datasets_url, headers=headers, auth=auth)
                    
                    if response.status_code == 200:
                        datasets = response.json()
                        if api_version == "v1":
                            existing = next((d for d in datasets.get("items", []) if d["name"] == dataset_name), None)
                        else:
                            existing = next((d for d in datasets if d["name"] == dataset_name), None)
                            
                        if existing:
                            print(f"Dataset '{dataset_name}' already exists, deleting it...")
                            if api_version == "v1":
                                delete_url = f"{url}/api/v1/datasets/{dataset_name}"
                            else:
                                delete_url = f"{url}/api/datasets/{dataset_name}"
                                
                            response = requests.delete(delete_url, headers=headers, auth=auth)
                            if response.status_code < 300:
                                print(f"✅ Deleted existing dataset")
                                time.sleep(1)  # Wait for deletion to complete
                            else:
                                print(f"⚠️ Failed to delete dataset: {response.status_code}")
                except Exception as e:
                    print(f"⚠️ Error checking existing datasets: {str(e)}")
                
                # Create dataset
                try:
                    response = requests.post(dataset_url, json=payload, headers=headers, auth=auth)
                    
                    if response.status_code == 200 or response.status_code == 201:
                        print(f"✅ Created dataset '{dataset_name}'")
                        
                        # Return the working authentication for adding records
                        return True, {"headers": headers, "auth": auth, "api_version": api_version}
                    else:
                        print(f"⚠️ Failed to create dataset: {response.status_code}")
                        print(response.text)
                except Exception as e:
                    print(f"⚠️ Error creating dataset: {str(e)}")
            
        except Exception as e:
            print(f"⚠️ Error testing authentication: {str(e)}")
    
    print("\n❌ All authentication methods failed.")
    return False, None

def add_sample_records(args, auth_info):
    """Add sample documents to the dataset"""
    url = args.url.rstrip("/")
    dataset_name = args.dataset
    api_version = auth_info["api_version"]
    headers = auth_info["headers"]
    auth = auth_info["auth"]
    
    success_count = 0
    
    for doc in SAMPLE_DOCUMENTS:
        try:
            # Format record according to API version
            if api_version == "v1":
                record_url = f"{url}/api/v1/datasets/{dataset_name}/TextClassification:records"
                record = {
                    "text": doc["text"],
                    "prediction": [[doc["label"], doc["confidence"]]],
                    "metadata": {
                        "confidence": doc["confidence"],
                        "source": "sample"
                    }
                }
            else:
                record_url = f"{url}/api/datasets/{dataset_name}/TextClassification:records"
                record = {
                    "text": doc["text"],
                    "prediction": [[doc["label"], doc["confidence"]]],
                    "metadata": {
                        "confidence": doc["confidence"],
                        "source": "sample"
                    }
                }
            
            # Add record
            response = requests.post(record_url, json=record, headers=headers, auth=auth)
            
            if response.status_code == 200 or response.status_code == 201:
                success_count += 1
            else:
                print(f"⚠️ Failed to add record: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error adding record: {str(e)}")
    
    print(f"\n✅ Added {success_count}/{len(SAMPLE_DOCUMENTS)} sample documents to Argilla")
    return success_count > 0

def main():
    args = parse_args()
    
    print("\n" + "="*50)
    print("🚀 SIMPLIFIED ARGILLA SETUP")
    print("="*50 + "\n")
    
    # Create dataset
    success, auth_info = create_argilla_dataset(args)
    
    if not success:
        print("❌ Failed to set up Argilla. Exiting.")
        sys.exit(1)
    
    # Add sample documents
    if add_sample_records(args, auth_info):
        print("\n✅ Successfully added sample documents to Argilla")
    else:
        print("\n⚠️ Failed to add sample documents, but dataset was created")
    
    print("\n" + "="*50)
    print("✅ ARGILLA SETUP COMPLETE!")
    print("="*50)
    print(f"\n🔗 Access Argilla at: {args.url}")
    print("\n💡 Use the working credentials from above to connect with your application")
    
    # Success
    sys.exit(0)

if __name__ == "__main__":
    main() 
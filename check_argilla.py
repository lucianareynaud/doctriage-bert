#!/usr/bin/env python3
"""
Simple script to test Argilla connection and authentication
"""

import requests
import sys

# Argilla configuration - Updated with correct values from container env
ARGILLA_API_URL = "http://localhost:6900"

# Multiple potential credentials to try
USERS = [
    {"username": "argilla", "password": "12345678"},  # From docker-compose.yml
    {"username": "admin", "password": "12345678"},    # From docker-compose.yml
    {"username": "admin", "password": "1234"},        # Default from container
    {"username": "argilla", "password": "1234"},      # Annotator with default password
    {"username": "owner", "password": "12345678"},    # Owner with custom password
    {"username": "owner", "password": "1234"}         # Owner with default password
]

API_KEYS = [
    "argilla.apikey",   # From docker-compose.yml and container default
    "admin.apikey",     # Potential admin key
    "team.apikey"       # From docker-compose.yml
]

def test_api_endpoint(path, method="get", auth=None, api_key=None, json_data=None):
    """Test an Argilla API endpoint with various auth methods"""
    url = f"{ARGILLA_API_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    if api_key:
        headers["X-API-KEY"] = api_key
    
    try:
        if method.lower() == "get":
            response = requests.get(url, auth=auth, headers=headers)
        elif method.lower() == "post":
            response = requests.post(url, auth=auth, headers=headers, json=json_data)
        else:
            print(f"Unsupported method: {method}")
            return None
            
        return response
    except Exception as e:
        print(f"Error with request to {url}: {e}")
        return None

def check_argilla_auth():
    """Test Argilla API authentication"""
    print(f"Testing connection to {ARGILLA_API_URL}...")
    
    # Check if Argilla is running
    try:
        response = requests.get(f"{ARGILLA_API_URL}/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Argilla server is running")
        else:
            # Try alternative endpoint
            response = requests.get(f"{ARGILLA_API_URL}/api/v1/status", timeout=5)
            if response.status_code == 200:
                print("✅ Argilla server is running (v1 API)")
            else:
                print(f"❌ Argilla server returned status code {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Error connecting to Argilla server: {e}")
        return False
    
    # Try all API key combinations
    for api_key in API_KEYS:
        print(f"\nTrying API key: {api_key}")
        
        # Try v0 API
        response = test_api_endpoint("/api/datasets", api_key=api_key)
        if response and response.status_code == 200:
            print(f"✅ Authentication successful with API key (v0 API)")
            return True
            
        # Try v1 API
        response = test_api_endpoint("/api/v1/me", api_key=api_key)
        if response and response.status_code == 200:
            print(f"✅ Authentication successful with API key (v1 API)")
            
            # Try to list datasets
            response = test_api_endpoint("/api/v1/datasets", api_key=api_key)
            if response and response.status_code == 200:
                try:
                    datasets = response.json()
                    dataset_count = len(datasets.get("items", []))
                    print(f"✅ Found {dataset_count} datasets.")
                    return True
                except:
                    print("⚠️ Could parse dataset response, but authentication worked")
                    return True
    
    # Try all username/password combinations
    for user in USERS:
        username = user["username"]
        password = user["password"]
        auth = (username, password)
        
        print(f"\nTrying Basic Auth: {username}:{password}")
        
        # Try v0 API
        response = test_api_endpoint("/api/datasets", auth=auth)
        if response and response.status_code == 200:
            print(f"✅ Authentication successful with Basic Auth (v0 API)")
            print(f"Use username: {username}, password: {password}")
            return True
            
        # Try v1 API
        response = test_api_endpoint("/api/v1/datasets", auth=auth)
        if response and response.status_code == 200:
            print(f"✅ Authentication successful with Basic Auth (v1 API)")
            print(f"Use username: {username}, password: {password}")
            return True
    
    print("\n❌ All authentication methods failed.")
    print("Check Docker configuration and try restarting the Argilla container.")
    return False

if __name__ == "__main__":
    success = check_argilla_auth()
    sys.exit(0 if success else 1) 
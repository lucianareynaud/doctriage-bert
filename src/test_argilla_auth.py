#!/usr/bin/env python
"""
Test script to check Argilla version and authentication methods
"""

import requests
import json
from requests.auth import HTTPBasicAuth
import argilla as rg

ARGILLA_URL = "http://localhost:6900"

def test_server_info():
    """Check if Argilla server is running and get its version"""
    try:
        # Try to get info from the base URL
        response = requests.get(ARGILLA_URL)
        if response.status_code == 200:
            print(f"Server is running, status code: {response.status_code}")
            # Try to extract client version from HTML
            html = response.text
            if "clientVersion" in html:
                start_idx = html.find("clientVersion") + 15  # Length of "clientVersion":"
                end_idx = html.find('"', start_idx)
                version = html[start_idx:end_idx]
                print(f"Argilla client version: {version}")
        else:
            print(f"Server returned status code: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to server: {str(e)}")

def test_api_version():
    """Test which API version is available"""
    endpoints = [
        "/api/status",
        "/api/health",
        "/api/v1/me",
        "/api/v1/status",
        "/api/v1/workspaces"
    ]
    
    for endpoint in endpoints:
        try:
            url = f"{ARGILLA_URL}{endpoint}"
            response = requests.get(url)
            print(f"Endpoint {endpoint}: Status {response.status_code}")
            if response.status_code == 401:  # Unauthorized but exists
                print(f"  Endpoint exists but requires authentication")
            elif response.status_code == 200:
                print(f"  Response: {response.text[:100]}...")
        except Exception as e:
            print(f"Error for {endpoint}: {str(e)}")

def test_auth_methods():
    """Test different authentication methods"""
    endpoint = f"{ARGILLA_URL}/api/v1/me"
    
    # Method 1: Basic Auth with username/password
    usernames = ["argilla", "admin", "team"]
    passwords = ["12345678", "1234", "argilla"]
    
    print("\nTesting Basic Auth combinations:")
    for username in usernames:
        for password in passwords:
            try:
                auth = HTTPBasicAuth(username, password)
                response = requests.get(endpoint, auth=auth)
                if response.status_code == 200:
                    print(f"✅ Success with user {username} / pass {password}")
                    print(f"  Response: {response.text}")
                else:
                    print(f"❌ Failed with user {username} / pass {password}")
            except Exception as e:
                print(f"Error with {username}/{password}: {str(e)}")
    
    # Method 2: API Key in headers
    api_keys = ["argilla.apikey", "team.apikey", "argilla:12345678", "admin:1234"]
    
    print("\nTesting API Key combinations:")
    for api_key in api_keys:
        try:
            headers = {"X-API-KEY": api_key}
            response = requests.get(endpoint, headers=headers)
            if response.status_code == 200:
                print(f"✅ Success with API key {api_key}")
                print(f"  Response: {response.text}")
            else:
                print(f"❌ Failed with API key {api_key}")
        except Exception as e:
            print(f"Error with API key {api_key}: {str(e)}")

if __name__ == "__main__":
    print("=== Testing Argilla Server ===")
    print(f"Using Argilla URL: {ARGILLA_URL}")
    print(f"Installed Argilla version: {rg.__version__}")
    print("\n=== Server Info ===")
    test_server_info()
    print("\n=== API Endpoints ===")
    test_api_version()
    print("\n=== Authentication ===")
    test_auth_methods() 
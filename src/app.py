"""
Streamlit frontend for DocTriage-BERT document classification.
"""

import os
import streamlit as st
import pandas as pd
import requests
import json
import time
import io
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
UPLOAD_FOLDER = "temp_uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Streamlit page config
st.set_page_config(
    page_title="DocTriage-BERT",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
        margin-bottom: 1rem;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .error-text {
        color: #dc3545;
        font-weight: bold;
    }
    .info-text {
        color: #17a2b8;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def predict_text(text):
    """Make a prediction for text."""
    response = requests.post(
        f"{API_URL}/predict/text",
        data={"text": text}
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None

def predict_file(file):
    """Make a prediction for a single file."""
    files = {"file": (file.name, file, "application/pdf")}
    response = requests.post(
        f"{API_URL}/predict/file",
        files=files
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None

def predict_batch(files):
    """Submit a batch prediction job."""
    files_data = [("files", (file.name, file, "application/pdf")) for file in files]
    response = requests.post(
        f"{API_URL}/predict/batch",
        files=files_data
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None

def get_job_status(job_id):
    """Get the status of a batch job."""
    response = requests.get(f"{API_URL}/jobs/{job_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None

def display_prediction_result(result, display_text=True):
    """Display a prediction result in a formatted way."""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"#### {result['filename']}")
        st.markdown(f"**Prediction:** <span class='{'success-text' if result['prediction'] == 'reports' else 'info-text'}'>{result['prediction'].upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {result['confidence']:.2%}")
        
        if display_text and result['text_preview']:
            with st.expander("Text Preview"):
                st.text(result['text_preview'])
    
    with col2:
        # Create a simple gauge chart for confidence
        fig, ax = plt.subplots(figsize=(3, 3))
        
        # Create pie chart showing confidence
        sizes = [result['confidence'], 1-result['confidence']]
        labels = ['', '']
        colors = ['#1E88E5' if result['prediction'] == 'reports' else '#9C27B0', '#f0f0f0']
        
        ax.pie(sizes, labels=labels, colors=colors, startangle=90, wedgeprops=dict(width=0.3))
        ax.text(0, 0, f"{result['confidence']:.0%}", ha='center', va='center', fontsize=24)
        
        # Add title based on prediction
        plt.title(result['prediction'].upper(), fontsize=16)
        
        # Make the pie chart a circle
        ax.set_aspect('equal')
        
        st.pyplot(fig)

def display_batch_results(results):
    """Display batch prediction results with charts."""
    if not results:
        st.warning("No results to display")
        return
    
    # Prepare dataframe
    df = pd.DataFrame([{
        'filename': r['filename'],
        'prediction': r['prediction'],
        'confidence': r['confidence']
    } for r in results])
    
    # Display summary charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Count by prediction
        st.subheader("Documents by Type")
        fig, ax = plt.subplots(figsize=(5, 5))
        counts = df['prediction'].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
              colors=['#1E88E5', '#9C27B0'] if 'reports' in counts.index else ['#9C27B0', '#1E88E5'])
        ax.axis('equal')
        st.pyplot(fig)
    
    with col2:
        # Confidence distribution
        st.subheader("Confidence Distribution")
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.histplot(df['confidence'], bins=10, ax=ax)
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    # Display results table
    st.subheader("Results Table")
    st.dataframe(df)
    
    # Display individual results
    st.subheader("Individual Results")
    for result in results:
        with st.expander(f"{result['filename']} - {result['prediction'].upper()}"):
            display_prediction_result(result)
    
def main():
    """Main Streamlit application."""
    st.markdown("<h1 class='main-header'>DocTriage-BERT</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Document Classification: Reports vs. Regulations</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    DocTriage-BERT classifies PDF documents as either **reports** or **regulations** using a fine-tuned language model.
    
    **Features:**
    - Upload single or multiple PDFs
    - Enter text for classification
    - View detailed prediction results
    """)
    
    # API connection status
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API connected")
    else:
        st.sidebar.error("❌ API not available")
        st.error("The classification API is not available. Please check the server status.")
        return
    
    # Tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["Single Document", "Batch Processing", "Text Classification"])
    
    # Tab 1: Single Document
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Upload a PDF document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="single_file")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing document..."):
                result = predict_file(uploaded_file)
                
            if result:
                display_prediction_result(result)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 2: Batch Processing
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Upload multiple PDF documents")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, key="batch_files")
        
        if uploaded_files:
            if st.button("Process Batch", key="batch_button"):
                with st.spinner("Submitting batch job..."):
                    batch_response = predict_batch(uploaded_files)
                
                if batch_response:
                    job_id = batch_response['job_id']
                    st.info(f"Batch job submitted with ID: {job_id}")
                    
                    # Create placeholders for progress
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    results_area = st.empty()
                    
                    # Poll for job status
                    complete = False
                    while not complete:
                        time.sleep(1)  # Poll every second
                        status_data = get_job_status(job_id)
                        
                        if status_data:
                            progress = status_data['progress']
                            status = status_data['status']
                            progress_bar.progress(progress)
                            progress_text.text(f"Status: {status} - Progress: {progress:.0%}")
                            
                            if status == "completed":
                                complete = True
                                progress_bar.progress(1.0)
                                progress_text.success("Batch processing complete!")
                                
                                # Display results
                                with results_area.container():
                                    display_batch_results(status_data['results'])
                                break
                            
                            elif status == "failed":
                                complete = True
                                progress_text.error(f"Batch processing failed: {status_data.get('error', 'Unknown error')}")
                                break
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Text Classification
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Classify Text")
        text_input = st.text_area("Enter text to classify", height=300)
        
        if st.button("Classify Text", key="text_button") and text_input:
            with st.spinner("Analyzing text..."):
                result = predict_text(text_input)
                
            if result:
                display_prediction_result(result, display_text=False)
                
                # Display text preview in an expander
                with st.expander("Text Preview"):
                    st.text(text_input[:500] + "..." if len(text_input) > 500 else text_input)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
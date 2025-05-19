# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with specific versions
RUN pip install --no-cache-dir \
    evaluate==0.4.1 \
    transformers==4.35.2 \
    datasets==2.15.0 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.1 \
    peft==0.6.2 \
    torch==2.1.1 \
    scikit-learn==1.3.2 \
    nlpaug==1.1.11 \
    -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads and outputs
RUN mkdir -p uploads outputs data

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/outputs/distil-lora-4bit
ENV UPLOAD_DIR=/app/uploads

# Default command - use this for API only
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"] 
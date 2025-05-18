# DocTriage-BERT

**DistilBERT + LoRA + 4-bit** classifier for document domain identification (reports vs. regulations), optimized for resource-constrained environments.

## Overview

DocTriage-BERT ingests PDFs by domain, extracts text using Tesseract OCR, shards into Parquet, and fine-tunes a lightweight DistilBERT classifier with parameter-efficient LoRA adapters and 4-bit quantization. Includes end-to-end pipeline:

1. **Ingestion & OCR** (`src/ingest_domains.py`)
2. **Parquet Sharding** (`data/*.parquet`)
3. **Training** (`src/train.py`)
4. **Evaluation** (test split)
5. **Human-in-the-loop Review** with Argilla
6. **Inference API & Streamlit UI**

---

## Quickstart

### Using the all-in-one script

The easiest way to set up and run the entire application is to use the provided `start.sh` script:

```bash
./start.sh
```

This script will:
1. Install dependencies (Colima, Docker)
2. Create necessary directories
3. Ingest and process documents
4. Train the model or select an existing model
5. Start all services

### Manual Setup

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Prepare Data

Place domain-specific PDFs under:

```
raw/reports/*.pdf
raw/regulations/*.pdf
```

Then ingest:

```bash
python src/ingest_domains.py \
  --input-dir raw/ \
  --output-dir data/ \
  --domains reports regulations
```

#### 3. Split Train/Test

```bash
mkdir -p raw/test/{reports,regulations}
# move one PDF per domain into raw/test/
python src/ingest_domains.py \
  --input-dir raw/ \
  --output-dir data/train_valid/ \
  --domains reports regulations
python src/ingest_domains.py \
  --input-dir raw/test/ \
  --output-dir data/test/ \
  --domains reports regulations
```

#### 4. Train & Evaluate

```bash
python src/train.py \
  --model distilbert-base-uncased \
  --output_dir outputs/distil-lora-4bit \
  --epochs 3 \
  --batch_size 4 \
  --lr 2e-5
```

#### 5. Start Services

```bash
docker-compose up
```

#### 6. Access Services

- API: http://localhost:8000
- Streamlit UI: http://localhost:8501
- Argilla: http://localhost:6900

---

## Project Structure

```
├── raw/                  # source PDFs
├── data/                 # processed Parquet shards
│   ├── train_valid/      # train+validation shards
│   └── test/             # test shards
├── src/
│   ├── ingest_domains.py # PDF processing with Tesseract OCR
│   ├── train.py          # model training
│   ├── api.py            # FastAPI backend
│   ├── app.py            # Streamlit UI
│   ├── log_for_review.py # Argilla integration
│   └── evaluate_review.py
├── outputs/              # model checkpoints
├── uploads/              # user uploaded files
├── start.sh              # all-in-one setup and deployment script
├── docker-compose.yml    # multi-container configuration
├── Dockerfile            # container definition
├── requirements.txt
└── README.md
```


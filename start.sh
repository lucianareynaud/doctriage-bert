#!/bin/bash

# Master script for DocTriage-BERT: setup, ingest, train, and deploy
set -e

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}DocTriage-BERT: All-in-one Setup and Deployment${NC}\n"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew is not installed! Please install Homebrew first:${NC}"
    echo -e "${YELLOW}/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${NC}"
    exit 1
fi

# Check if Colima is installed, if not install it
if ! command -v colima &> /dev/null; then
    echo -e "${YELLOW}Colima not found. Installing...${NC}"
    brew install colima
fi

# Check if Docker is installed, if not install it
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker not found. Installing...${NC}"
    brew install docker
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Docker Compose not found. Installing...${NC}"
    brew install docker-compose
fi

# Check if Colima is running
if ! colima status &> /dev/null; then
    echo -e "${YELLOW}Starting Colima...${NC}"
    colima start
    echo -e "${GREEN}Colima started successfully${NC}"
fi

# Create required directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p data/train_valid data/test outputs uploads raw/{reports,regulations} raw/test/{reports,regulations}

# Check if there are PDFs in the raw directory
reports_count=$(find raw/reports -name "*.pdf" 2>/dev/null | wc -l)
regulations_count=$(find raw/regulations -name "*.pdf" 2>/dev/null | wc -l)

# Check if ingestion is needed
need_ingestion=true
train_files=$(find data/train_valid -name "*.parquet" 2>/dev/null | wc -l)
test_files=$(find data/test -name "*.parquet" 2>/dev/null | wc -l)

if [ "$train_files" -gt 0 ] && [ "$test_files" -gt 0 ]; then
    echo -e "${GREEN}Found existing processed data files.${NC}"
    read -p "Skip ingestion and use existing data? [Y/n]: " skip_ingestion
    if [[ "$skip_ingestion" != "n" && "$skip_ingestion" != "N" ]]; then
        need_ingestion=false
    fi
fi

# Handle ingestion if needed
if [ "$need_ingestion" = true ]; then
    if [ "$reports_count" -eq 0 ] || [ "$regulations_count" -eq 0 ]; then
        echo -e "${RED}Warning: One or more document categories are missing PDF files.${NC}"
        echo -e "${YELLOW}Please ensure you have PDF files in:${NC}"
        echo -e "  - raw/reports/ ($reports_count files found)"
        echo -e "  - raw/regulations/ ($regulations_count files found)"
        read -p "Continue without some document types? [y/N]: " continue_without_docs
        if [[ "$continue_without_docs" != "y" && "$continue_without_docs" != "Y" ]]; then
            echo -e "${YELLOW}Please add PDF files to the raw directories and run this script again.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}Found $reports_count reports and $regulations_count regulations.${NC}"
    fi

    # Move some documents to the test set if test is empty
    test_reports=$(find raw/test/reports -name "*.pdf" 2>/dev/null | wc -l)
    test_regulations=$(find raw/test/regulations -name "*.pdf" 2>/dev/null | wc -l)

    if [ "$test_reports" -eq 0 ] && [ "$reports_count" -gt 1 ]; then
        FIRST_REPORT=$(find raw/reports -name "*.pdf" | head -1)
        cp "$FIRST_REPORT" raw/test/reports/
        echo -e "${GREEN}Copied 1 report to test set.${NC}"
    fi
    
    if [ "$test_regulations" -eq 0 ] && [ "$regulations_count" -gt 1 ]; then
        FIRST_REGULATION=$(find raw/regulations -name "*.pdf" | head -1)
        cp "$FIRST_REGULATION" raw/test/regulations/
        echo -e "${GREEN}Copied 1 regulation to test set.${NC}"
    fi

    # Build Docker images if not already built
    if ! docker images | grep -q doctriage-bert-api; then
        echo -e "${YELLOW}Building Docker images...${NC}"
        docker-compose build
        echo -e "${GREEN}Docker images built successfully.${NC}"
    fi

    # Run the ingest command in Docker
    echo -e "${YELLOW}Ingesting training documents...${NC}"
    docker run --rm -v "$(pwd):/app" doctriage-bert-api:latest python src/ingest_domains.py \
        --input-dir /app/raw \
        --output-dir /app/data/train_valid \
        --domains reports regulations

    echo -e "${YELLOW}Ingesting test documents...${NC}"
    docker run --rm -v "$(pwd):/app" doctriage-bert-api:latest python src/ingest_domains.py \
        --input-dir /app/raw/test \
        --output-dir /app/data/test \
        --domains reports regulations
    
    echo -e "${GREEN}Ingestion complete!${NC}"
fi

# Check if training is needed
need_training=true
# Find all model directories
model_dirs=$(find outputs -maxdepth 1 -type d -not -path "outputs" 2>/dev/null)
model_path=""

if [ -n "$model_dirs" ]; then
    echo -e "${GREEN}Found existing model directories:${NC}"
    # Display available models with numbers
    i=1
    for dir in $model_dirs; do
        echo -e "$i) $dir"
        i=$((i+1))
    done
    echo -e "$i) Train a new model"
    
    # Prompt user to select a model or train a new one
    read -p "Select an option [${i}]: " model_option
    model_option=${model_option:-$i}
    
    if [ "$model_option" -lt "$i" ]; then
        # User selected an existing model
        selected_model=$(echo "$model_dirs" | sed -n "${model_option}p")
        model_path="/app/$selected_model"
        need_training=false
        echo -e "${GREEN}Selected model: $selected_model${NC}"
    fi
fi

# Handle training if needed
if [ "$need_training" = true ]; then
    echo -e "\n${BOLD}${BLUE}Model Training${NC}\n"
    
    # Ask for training parameters
    read -p "Number of epochs [3]: " epochs
    epochs=${epochs:-3}

    read -p "Batch size [4]: " batch_size
    batch_size=${batch_size:-4}

    read -p "Learning rate [2e-5]: " learning_rate
    learning_rate=${learning_rate:-2e-5}

    read -p "Model name [distilbert-base-uncased]: " model_name
    model_name=${model_name:-distilbert-base-uncased}

    # Create the output directory
    output_dir="outputs/distil-lora-4bit-$(date +%Y%m%d-%H%M%S)"
    mkdir -p $output_dir
    model_path="/app/$output_dir"

    echo -e "${YELLOW}Starting training with the following parameters:${NC}"
    echo -e "- Model: ${BOLD}$model_name${NC}"
    echo -e "- Epochs: ${BOLD}$epochs${NC}"
    echo -e "- Batch size: ${BOLD}$batch_size${NC}"
    echo -e "- Learning rate: ${BOLD}$learning_rate${NC}"
    echo -e "- Output directory: ${BOLD}$output_dir${NC}"

    # Run the training command in Docker (defaults to CPU mode for MacBook compatibility)
    echo -e "\n${YELLOW}Training model (optimized for MacBook)...${NC}"
    echo -e "${GREEN}Using CPU-optimized mode with gradient checkpointing and accumulation...${NC}"
    
    docker run --rm -v "$(pwd):/app" doctriage-bert-api:latest python src/train.py \
        --model $model_name \
        --output_dir $model_path \
        --epochs $epochs \
        --batch_size $batch_size \
        --lr $learning_rate \
        --train_data /app/data/train_valid \
        --test_data /app/data/test \
        --gradient_accumulation_steps 4

    echo -e "${GREEN}Training complete!${NC}"
fi

# Update the MODEL_PATH in docker-compose.yml
echo -e "${YELLOW}Updating MODEL_PATH in docker-compose.yml...${NC}"
sed -i '' "s|MODEL_PATH=.*|MODEL_PATH=$model_path|g" docker-compose.yml
echo -e "${GREEN}Updated MODEL_PATH to: $model_path${NC}"

# Remove the authentication attempt before starting services
# USE_ARGILLA_API is set here but used later after service is running
USE_ARGILLA_API=true
ARGILLA_API_URL="http://localhost:6900"
USERNAME="argilla"
PASSWORD="12345678"

# Start the services
echo -e "\n${BOLD}${BLUE}Starting DocTriage-BERT Services${NC}\n"
echo -e "${YELLOW}Starting with docker-compose...${NC}"
docker-compose up -d

# Wait for Argilla to be ready
echo -e "${YELLOW}Waiting for Argilla to be ready...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s --head --request GET http://localhost:6900/api/v1/status | grep "200 OK" > /dev/null; then
        echo -e "${GREEN}Argilla is up and running!${NC}"
        break
    else
        echo -n "."
        RETRY_COUNT=$((RETRY_COUNT+1))
        sleep 4
    fi
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "\n${RED}Timed out waiting for Argilla to start.${NC}"
        echo -e "${YELLOW}You can continue using the API and UI, but Argilla data annotation won't be available.${NC}"
        break
    fi
done

# Additional safety wait time - sometimes Argilla needs a moment after the API is responsive
sleep 10

# Now that Argilla is running, attempt to authenticate
if [ "$USE_ARGILLA_API" = true ]; then
    echo -e "${YELLOW}Setting up Argilla and annotation pipeline for DocTriage-BERT...${NC}"
    
    # Try to use the setup_argilla_simple.py script which handles authentication internally
    python setup_argilla_simple.py --url $ARGILLA_API_URL --username $USERNAME --password $PASSWORD
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Argilla setup completed successfully!${NC}"
    else
        echo -e "${YELLOW}Trying alternative Argilla setup method...${NC}"
        # Use the check_argilla.py script to verify credentials
        python check_argilla.py
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Argilla credentials verified!${NC}"
        else
            echo -e "${RED}❌ Failed to verify Argilla credentials${NC}"
            echo -e "${YELLOW}You can still access Argilla UI at: http://localhost:6900${NC}"
            echo -e "${YELLOW}Login with username: argilla / password: 12345678${NC}"
        fi
    fi
else
    # Setup Argilla with sample data using the previous method
    echo -e "${YELLOW}Setting up Argilla with sample data...${NC}"
    python src/argilla_setup.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Argilla setup completed successfully!${NC}"
    else
        echo -e "${RED}Argilla setup encountered issues.${NC}"
        echo -e "${YELLOW}You can still access Argilla UI at: http://localhost:6900${NC}"
    fi
fi

# Show API endpoint
echo -e "\n${BOLD}${GREEN}DocTriage-BERT services are running!${NC}"
echo -e "${BLUE}API endpoint: http://localhost:8181/docs${NC}"
echo -e "${BLUE}Streamlit UI: http://localhost:8501${NC}"
echo -e "${BLUE}Argilla annotation tool: http://localhost:6900${NC}"

# Display links for manual exploration
echo -e "\n${BOLD}${BLUE}Explore the data and models:${NC}"
echo -e "1. ${YELLOW}Use the Streamlit UI:${NC} http://localhost:8501"
echo -e "2. ${YELLOW}Explore the API docs:${NC} http://localhost:8181/docs"
echo -e "3. ${YELLOW}Annotate documents:${NC} http://localhost:6900"

echo -e "\n${BOLD}${BLUE}To stop all services:${NC}"
echo -e "${YELLOW}docker-compose down${NC}"

echo -e "\n${GREEN}Setup complete!${NC}" 
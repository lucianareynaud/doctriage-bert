#!/bin/bash

# Script to start Argilla and set up the annotation pipeline
set -e

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}DocTriage-BERT: Annotation Pipeline Setup${NC}\n"

# Create required directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p webhooks data/annotations

# Check if Docker is running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed! Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is running
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed! Please install Docker Compose first.${NC}"
    exit 1
fi

# Start Docker containers if not already running
if ! docker ps | grep -q doctriage-argilla; then
    echo -e "${YELLOW}Starting Docker containers...${NC}"
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
            sleep 3
        fi
        
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo -e "\n${RED}Timed out waiting for Argilla to start.${NC}"
            echo -e "${YELLOW}Please check Docker logs: docker logs doctriage-argilla${NC}"
            exit 1
        fi
    done
else
    echo -e "${GREEN}Docker containers are already running.${NC}"
fi

# Install required dependencies for the annotation pipeline
echo -e "${YELLOW}Installing required dependencies...${NC}"
if [ -f webhooks/requirements.txt ]; then
    pip install -r webhooks/requirements.txt
else
    pip install requests
fi

# Set up Argilla and the annotation pipeline
echo -e "${YELLOW}Setting up Argilla and the annotation pipeline...${NC}"
python src/argilla_setup.py

# Create a shortcut to process annotations
echo -e "${YELLOW}Creating annotation processing shortcut...${NC}"
cat > process_annotations.sh << EOL
#!/bin/bash

# Process annotations and retrain the model
echo "Processing annotations from Argilla..."
python webhooks/process_annotations.py --min-annotations 5

# Check exit status
if [ \$? -eq 0 ]; then
  echo "✅ Successfully processed annotations"
else
  echo "❌ Failed to process annotations"
fi
EOL

chmod +x process_annotations.sh

echo -e "\n${BOLD}${GREEN}Annotation pipeline setup complete!${NC}"
echo -e "${BLUE}You can now:${NC}"
echo -e "1. Access Argilla at: ${GREEN}http://localhost:6900${NC}"
echo -e "   Login with username: ${BOLD}argilla${NC}, password: ${BOLD}12345678${NC}"
echo -e "2. Process annotations by running: ${YELLOW}./process_annotations.sh${NC}"
echo -e "3. Set up automatic retraining by adding the cron job shown above"

echo -e "\n${BOLD}${BLUE}To stop all services:${NC}"
echo -e "${YELLOW}docker-compose down${NC}"

echo -e "\n${GREEN}Setup complete!${NC}" 
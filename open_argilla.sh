#!/bin/bash

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}Opening Argilla with auto-login...${NC}\n"

# Check if containers are running
if ! docker ps | grep -q doctriage-argilla; then
    echo -e "${RED}Argilla container is not running!${NC}"
    echo -e "${YELLOW}Starting services...${NC}"
    docker-compose up -d
    
    # Wait for Argilla to be ready
    echo -e "${YELLOW}Waiting for Argilla to be ready...${NC}"
    for i in {1..15}; do
        if curl -s -f http://localhost:6900 > /dev/null; then
            echo -e "${GREEN}Argilla is up and running!${NC}"
            break
        fi
        echo -n "."
        sleep 2
        
        if [ $i -eq 15 ]; then
            echo -e "\n${RED}Timed out waiting for Argilla to start.${NC}"
            echo -e "${YELLOW}Please check the container logs: ${NC}docker logs doctriage-argilla"
            exit 1
        fi
    done
fi

# Try to open the browser
if command -v open &> /dev/null; then
    # macOS
    echo -e "${GREEN}Opening Argilla in browser...${NC}"
    open argilla_login.html
elif command -v xdg-open &> /dev/null; then
    # Linux
    echo -e "${GREEN}Opening Argilla in browser...${NC}"
    xdg-open argilla_login.html
elif command -v start &> /dev/null; then
    # Windows
    echo -e "${GREEN}Opening Argilla in browser...${NC}"
    start argilla_login.html
else
    echo -e "${YELLOW}Could not automatically open browser.${NC}"
    echo -e "${GREEN}Please open this URL:${NC} http://localhost:6900"
    echo -e "${GREEN}Login credentials:${NC}"
    echo -e "  Username: ${BOLD}argilla${NC}"
    echo -e "  Password: ${BOLD}12345678${NC}"
fi

echo -e "\n${GREEN}Argilla is running at:${NC} http://localhost:6900" 
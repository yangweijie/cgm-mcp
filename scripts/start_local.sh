#!/bin/bash

# CGM MCP Server - Local Model Startup Script
# Supports Ollama and LM Studio local models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
PROVIDER="ollama"
MODEL=""
CONFIG_FILE=""
PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --provider PROVIDER   LLM provider (ollama, lmstudio) [default: ollama]"
            echo "  --model MODEL         Model name"
            echo "  --config FILE         Configuration file"
            echo "  --port PORT           Server port [default: 8000]"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --provider ollama --model deepseek-coder:6.7b"
            echo "  $0 --provider lmstudio --model deepseek-coder-6.7b-instruct"
            echo "  $0 --config config.local.json"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Starting CGM MCP Server with local models..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set configuration based on provider
if [ -z "$CONFIG_FILE" ]; then
    case $PROVIDER in
        ollama)
            CONFIG_FILE="config.local.json"
            if [ -z "$MODEL" ]; then
                MODEL="deepseek-coder:6.7b"
            fi
            ;;
        ollama_cloud)
            CONFIG_FILE="config.ollama_cloud.json"
            if [ -z "$MODEL" ]; then
                MODEL="llama3"
            fi
            ;;
        lmstudio)
            CONFIG_FILE="config.lmstudio.json"
            if [ -z "$MODEL" ]; then
                MODEL="deepseek-coder-6.7b-instruct"
            fi
            ;;
        *)
            print_error "Unsupported provider: $PROVIDER"
            exit 1
            ;;
    esac
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Check provider service
check_service() {
    case $PROVIDER in
        ollama)
            if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                print_error "Ollama service not running. Please start Ollama first:"
                print_error "  ollama serve"
                exit 1
            fi

            # Check if model exists
            if ! ollama list | grep -q "$MODEL"; then
                print_warning "Model $MODEL not found. Downloading..."
                ollama pull "$MODEL"
            fi
            ;;
        ollama_cloud)
            # For Ollama Cloud, we just check if API key is set
            if [ -z "$CGM_LLM_API_KEY" ] && [ -z "$CGM_CLOUD_API_KEY" ]; then
                API_KEY=$(grep -E 'api_key' "$CONFIG_FILE" | head -1 | cut -d'"' -f4)
                if [ -z "$API_KEY" ] || [ "$API_KEY" = "your-ollama-cloud-api-key" ]; then
                    print_warning "No API key provided for Ollama Cloud. Please set CGM_LLM_API_KEY or update the config file."
                fi
            fi
            ;;
        lmstudio)
            if ! curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
                print_error "LM Studio service not running. Please start LM Studio server first."
                exit 1
            fi
            ;;
    esac
}

print_status "Checking $PROVIDER service..."
check_service

# Set environment variables
export CGM_LLM_PROVIDER="$PROVIDER"
export CGM_LLM_MODEL="$MODEL"
export CGM_SERVER_PORT="$PORT"

print_success "Service check passed"
print_status "Provider: $PROVIDER"
print_status "Model: $MODEL"
print_status "Config: $CONFIG_FILE"
print_status "Port: $PORT"

# Start the server
print_status "Starting CGM MCP Server..."
python main.py --config "$CONFIG_FILE" --log-level INFO

print_success "CGM MCP Server started successfully!"

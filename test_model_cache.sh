#!/bin/bash

# Local model cache verification test script
# Used to validate model cache locally before deployment

set -e

# Resolve script directory to get a reliable .env path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "LOCAL MODEL CACHE VERIFICATION TEST"
echo ""

# Validate .env
ENV_FILE="$SCRIPT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found in $SCRIPT_DIR!"
    echo "Please create .env file with required variables:"
    echo "  - GEMINI_API_KEY"
    echo "  - PINECONE_API_KEY"
    echo "  - HF_TOKEN"
    exit 1
fi

echo " .env file found at $ENV_FILE"

# Load variables
source "$ENV_FILE"

# Validate required variables
echo ""
echo "Checking environment variables..."
missing_vars=0

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set"
    missing_vars=1
else
    echo "OK: GEMINI_API_KEY is set"
fi

if [ -z "$PINECONE_API_KEY" ]; then
    echo "ERROR: PINECONE_API_KEY not set"
    missing_vars=1
else
    echo "OK: PINECONE_API_KEY is set"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set"
    missing_vars=1
else
    echo "OK: HF_TOKEN is set"
fi

if [ $missing_vars -eq 1 ]; then
    echo ""
    echo "ERROR: Some environment variables are missing!"
    exit 1
fi

# Validate Python
echo ""
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    exit 1
fi
echo "OK: Python3 found: $(python3 --version)"

# Validate required packages
echo ""
echo "Checking required packages..."
python3 -c "import huggingface_hub; print('OK: huggingface_hub installed')" || {
    echo "ERROR: huggingface_hub not installed"
    exit 1
}

python3 -c "import transformers; print('OK: transformers installed')" || {
    echo "ERROR: transformers not installed"
    exit 1
}

python3 -c "import sentence_transformers; print('OK: sentence_transformers installed')" || {
    echo "ERROR: sentence_transformers not installed"
    exit 1
}

python3 -c "import rapidocr_onnxruntime; print('OK: rapidocr_onnxruntime installed')" || {
    echo "ERROR: rapidocr_onnxruntime not installed"
    exit 1
}

python3 -c "import langchain_google_genai; print('OK: langchain_google_genai installed')" || {
    echo "ERROR: langchain_google_genai not installed"
    exit 1
}

python3 -c "import pinecone; print('OK: pinecone installed')" || {
    echo "ERROR: pinecone not installed"
    exit 1
}

# Run cache verification
echo ""

echo "Running model cache verification..."


python3 "$SCRIPT_DIR/verify_model_cache.py"

if [ $? -eq 0 ]; then
    echo ""
    echo "ALL CHECKS PASSED!"
    echo ""
    echo "You are ready to deploy!"
    echo "Run: ./deploy.sh"
    exit 0
else
    echo ""
    echo "VERIFICATION FAILED!"
    echo ""
    echo "Please fix the issues above and try again."
    exit 1
fi

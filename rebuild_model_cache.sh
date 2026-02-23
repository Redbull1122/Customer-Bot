#!/bin/bash

# Script to clear and re-download model cache
# Used to clear a corrupted cache and re-download models

set -e


echo "MODEL CACHE CLEANUP AND REBUILD"
echo ""

# Validate .env
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    exit 1
fi

source .env

# Validate HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set in .env"
    exit 1
fi

echo "WARNING: This will delete all cached models and re-download them."
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Clearing Hugging Face cache..."
HF_HOME="${HF_HOME:=$HOME/.cache/huggingface}"
if [ -d "$HF_HOME" ]; then
    rm -rf "$HF_HOME"
    echo "OK: Cleared: $HF_HOME"
else
    echo "WARNING: Directory not found: $HF_HOME"
fi

echo ""
echo "Clearing RapidOCR cache..."
RAPIDOCR_CACHE="$HOME/.cache/rapidocr"
if [ -d "$RAPIDOCR_CACHE" ]; then
    rm -rf "$RAPIDOCR_CACHE"
    echo "OK: Cleared: $RAPIDOCR_CACHE"
else
    echo "WARNING: Directory not found: $RAPIDOCR_CACHE"
fi

ONNX_CACHE="$HOME/.cache/onnxruntime"
if [ -d "$ONNX_CACHE" ]; then
    rm -rf "$ONNX_CACHE"
    echo "OK: Cleared: $ONNX_CACHE"
else
    echo "WARNING: Directory not found: $ONNX_CACHE"
fi

echo ""
echo "Re-downloading models..."
echo ""

# Run model download script
python3 download_models.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to download models!"
    exit 1
fi

echo ""
echo "Verifying downloaded models..."
echo ""

# Run verification
python3 verify_model_cache.py

if [ $? -eq 0 ]; then
    echo ""
    echo "CACHE REBUILD SUCCESSFUL."
    exit 0
else
    echo ""
    echo "VERIFICATION FAILED."
    exit 1
fi

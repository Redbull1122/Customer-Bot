#!/bin/bash

set -e
# --set-env-vars="PYTHONUNBUFFERED=1,GEMINI_API_KEY=$GEMINI_API_KEY,PINECONE_API_KEY=$PINECONE_API_KEY,HF_TOKEN=$HF_TOKEN"

PROJECT_ID="bot-hajster"
SERVICE_NAME="bot-hajster"

# Region selection
echo "Available regions:"
echo "1) us-central1 (Iowa)"
echo "2) europe-west1 (Belgium) - recommended for Ukraine"
echo "3) us-east1 (South Carolina)"
read -p "Select region (1-3): " region_choice

case $region_choice in
    1) REGION="us-central1" ;;
    2) REGION="europe-west1" ;;
    3) REGION="us-east1" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo "Using region: $REGION"

# Validate .env
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    exit 1
fi

source .env

# Validate required variables
missing_vars=0
[ -z "$GEMINI_API_KEY" ] && echo "ERROR: GEMINI_API_KEY missing" && missing_vars=1
[ -z "$PINECONE_API_KEY" ] && echo "ERROR: PINECONE_API_KEY missing" && missing_vars=1
[ -z "$HF_TOKEN" ] && echo "ERROR: HF_TOKEN missing" && missing_vars=1
[ -z "$PINECONE_INDEX" ] && echo "WARNING: PINECONE_INDEX missing (using default if set in code)"
if [ $missing_vars -eq 1 ]; then
    exit 1
fi

# Verify model cache before deployment
echo ""
echo "Verifying model cache before deployment..."

python3 verify_model_cache.py
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: DEPLOYMENT BLOCKED: Model cache verification failed!"
    echo "Please fix the issues above and try again."
    exit 1
fi
echo "OK: Model cache verification passed!"
echo ""

VERSION=$(date +%Y%m%d%H%M)
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME:$VERSION"

gcloud config set project $PROJECT_ID

echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

echo "Building Docker image using cloudbuild.yaml..."
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_VERSION="$VERSION",_HF_TOKEN="$HF_TOKEN",_GEMINI_API_KEY="$GEMINI_API_KEY",_PINECONE_API_KEY="$PINECONE_API_KEY" .

echo "Deploying to Cloud Run in $REGION..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_NAME" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 8Gi \
  --no-cpu-throttling \
  --cpu 4 \
  --concurrency 4 \
  --timeout 400 \
  --max-instances 5 \
  --min-instances 1 \
  --set-env-vars="PYTHONUNBUFFERED=1,GEMINI_API_KEY=$GEMINI_API_KEY,PINECONE_API_KEY=$PINECONE_API_KEY,HF_TOKEN=$HF_TOKEN,HF_HOME=/app/.cache/hf,TRANSFORMERS_CACHE=/app/.cache/hf,SENTENCE_TRANSFORMERS_HOME=/app/.cache/hf,PINECONE_INDEX=$PINECONE_INDEX"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format='value(status.url)')

echo ""
echo "Deployment completed!"
echo "Service URL: $SERVICE_URL"
echo "Region: $REGION"
echo "Memory: 8Gi"

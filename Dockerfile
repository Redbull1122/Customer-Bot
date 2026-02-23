FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Accept build arguments and set them as environment variables
ARG HF_TOKEN
ARG GEMINI_API_KEY
ARG PINECONE_API_KEY

ENV HF_TOKEN=${HF_TOKEN}
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV PINECONE_API_KEY=${PINECONE_API_KEY}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-6 \
    zlib1g \
    ca-certificates \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*
# ---- Workdir ----
WORKDIR /app

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ---- Pre-download models ----
COPY download_models.py .
COPY verify_model_cache.py .
RUN python download_models.py
RUN python verify_model_cache.py || (echo "ERROR: Model cache verification failed!" && exit 1)
RUN rm download_models.py verify_model_cache.py

# ---- App ----
COPY . .

# ---- Env ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/hf \
    TRANSFORMERS_CACHE=/app/.cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/hf \
    OPENCV_IO_ENABLE_OPENEXR=0 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg


RUN mkdir -p /app/.cache/hf


EXPOSE 8080

CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 300 --workers ${WEB_CONCURRENCY:-2} --preload api:app

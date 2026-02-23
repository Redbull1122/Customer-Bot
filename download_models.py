"""Pre-download model files to disk cache during Docker build without loading into memory"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



print("PRE-DOWNLOADING (CACHING) MODEL FILES...")

# Ensure a deterministic cache dir (Hugging Face + sentence-transformers)
HF_HOME = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
if not HF_HOME:
    # If running in a container (e.g. Cloud Run), default to /app/.cache/hf to match deploy.sh
    if os.path.exists("/app"):
        HF_HOME = "/app/.cache/hf"
    else:
        HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

# Set env vars to ensure consistency across libraries
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# --- HF Auth ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("FATAL: HF_TOKEN environment variable not set. Build cannot continue.")
    sys.exit(1)

# 1) Cache sentence-transformers model artifacts without keeping model in RAM
try:
    print("Caching sentence-transformers model artifacts...")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    
    # sentence-transformers models are hosted on HF Hub
    model_id = "intfloat/multilingual-e5-large"
    cache_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=HF_HOME,
        token=HF_TOKEN,
        local_dir_use_symlinks=False  # Use False for better cross-system compatibility
    )
    print(f"OK: Model downloaded to: {cache_dir}")

    # --- DIAGNOSTIC: List files in cache directory ---
    print(f"--- Listing files in {cache_dir} ---")
    try:
        for f in os.listdir(cache_dir):
            print(f"- {f}")
    except Exception as list_e:
        print(f"Could not list directory: {list_e}")
    print("------------------------------------")
    
    # Verify a key tokenizer file exists to confirm download integrity
    vocab_path = Path(cache_dir) / "tokenizer.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Key tokenizer file not found at {vocab_path}")
    print(f"OK: Tokenizer file verified: {vocab_path}")
    
    # Verify tokenizer can be loaded
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, cache_dir=HF_HOME)
        print("OK: Tokenizer loaded and verified successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    print("OK: sentence-transformers artifacts cached successfully.")
except Exception as e:
    print(f"FATAL: Failed to cache sentence-transformers artifacts: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2) Cache RapidOCR ONNX models to disk without holding objects
try:
    print("\nCaching RapidOCR ONNX models...")
    # RapidOCR downloads model files on first use; we can force download by importing and invoking a minimal path
    # without persisting the instantiated object.
    from rapidocr_onnxruntime import RapidOCR
    # Create then immediately delete to trigger download; avoid holding reference
    _tmp = RapidOCR()
    del _tmp
    print("OK: RapidOCR ONNX artifacts cached successfully.")
except Exception as e:
    print(f"WARNING: Failed to cache RapidOCR artifacts: {e}")
    # This part is not critical, so we can continue with a warning.

print("\nALL CRITICAL MODEL FILES CACHED TO DISK.")

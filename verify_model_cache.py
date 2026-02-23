"""
Verify that all required models are properly cached before deployment.
This script checks if models are accessible and not corrupted.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def verify_sentence_transformers_cache():
    """Verify sentence-transformers model is cached"""
    print("Verifying sentence-transformers model cache...")
    
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            print("ERROR: HF_TOKEN environment variable not set")
            return False
        
        model_id = "intfloat/multilingual-e5-large"
        
        # Try to download/verify the model
        try:
            cache_dir = snapshot_download(
                repo_id=model_id,
                token=HF_TOKEN,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"OK: Model cache directory: {cache_dir}")
            
            # Verify a key tokenizer file exists to confirm download integrity
            vocab_path = Path(cache_dir) / "tokenizer.json"
            if not vocab_path.exists():
                print(f"ERROR: Key tokenizer file not found at {vocab_path}")
                return False
            
            print(f"OK: Tokenizer file verified: {vocab_path}")
            
            # Try to load tokenizer to verify it's not corrupted
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
                print("OK: Tokenizer loaded successfully")
            except Exception as e:
                print(f"ERROR: Failed to load tokenizer: {e}")
                return False
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to download/verify model: {e}")
            return False
            
    except ImportError as e:
        print(f"ERROR: Required package not installed: {e}")
        return False


def verify_rapidocr_cache():
    """Verify RapidOCR ONNX models are cached"""
    print("\nVerifying RapidOCR ONNX models cache...")
    
    try:
        from rapidocr_onnxruntime import RapidOCR
        
        # Try to instantiate RapidOCR to trigger model loading
        try:
            ocr = RapidOCR()
            print("RapidOCR initialized successfully")
            
            # Check if model files exist
            # RapidOCR typically caches models in ~/.cache/rapidocr or similar
            cache_dirs = [
                Path.home() / ".cache" / "rapidocr",
                Path.home() / ".cache" / "onnxruntime",
                Path("/app/.cache/rapidocr"),
                Path("/app/.cache/onnxruntime"),
            ]
            
            model_found = False
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    model_files = list(cache_dir.glob("**/*.onnx"))
                    if model_files:
                        print(f"OK: Found {len(model_files)} ONNX model files in {cache_dir}")
                        model_found = True
                        break
            
            if not model_found:
                print("WARNING: Could not verify ONNX model files location")
                # This is a warning, not a fatal error
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize RapidOCR: {e}")
            return False
            
    except ImportError as e:
        print(f"ERROR: RapidOCR not installed: {e}")
        return False


def verify_gemini_api():
    """Verify Gemini API key is set"""
    print("\nVerifying Gemini API configuration...")
    
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return False
    
    print("OK: GEMINI_API_KEY is set")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Try to instantiate the model (this validates the API key format)
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            print("OK: ChatGoogleGenerativeAI initialized successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to initialize ChatGoogleGenerativeAI: {e}")
            return False
            
    except ImportError as e:
        print(f"ERROR: langchain-google-genai not installed: {e}")
        return False


def verify_pinecone_api():
    """Verify Pinecone API key is set"""
    print("\nVerifying Pinecone API configuration...")
    
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        print("ERROR: PINECONE_API_KEY environment variable not set")
        return False
    
    print("OK: PINECONE_API_KEY is set")
    
    try:
        from pinecone import Pinecone
        
        # Try to instantiate Pinecone client
        try:
            pc = Pinecone(api_key=pinecone_key)
            print("OK: Pinecone client initialized successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to initialize Pinecone client: {e}")
            return False
            
    except ImportError as e:
        print(f"ERROR: pinecone not installed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("MODEL CACHE VERIFICATION FOR DEPLOYMENT")
    
    checks = [
        ("Sentence Transformers", verify_sentence_transformers_cache),
        ("RapidOCR", verify_rapidocr_cache),
        ("Gemini API", verify_gemini_api),
        ("Pinecone API", verify_pinecone_api),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"FATAL ERROR in {check_name}: {e}")
            import traceback
            traceback.print_exc()
            results[check_name] = False

    print("VERIFICATION SUMMARY")

    
    all_passed = True
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    

    
    if all_passed:
        print("\nAll checks passed. Safe to deploy.")
        return 0
    else:
        print("\nSome checks failed! Deployment blocked.")
        print("\nPlease fix the issues above before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

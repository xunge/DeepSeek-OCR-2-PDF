import os

BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = os.getenv("CROP_MODE", "true").lower() in ("1", "true", "yes", "on")
MIN_CROPS = int(os.getenv("MIN_CROPS", "2"))
MAX_CROPS = int(os.getenv("MAX_CROPS", "6"))  # max:6
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "100"))  # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "64"))  # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = os.getenv("SKIP_REPEAT", "true").lower() in ("1", "true", "yes", "on")
MODEL_PATH = os.getenv("MODEL_PATH", "models/DeepSeek-OCR-2")
PROMPT = os.getenv("PROMPT", '<image>\nExtract the document content as markdown. For images, output [IMAGE DESCRIPTION: describe the image content in detail].')
# PROMPT = '<image>\nFree OCR.'
# .......

# Server configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

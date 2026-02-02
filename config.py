BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # max:6
MAX_CONCURRENCY = 100  # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64  # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = '/home/jiang/model/DeepSeek-OCR-2/'  # change to your model path
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# .......

# Server configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

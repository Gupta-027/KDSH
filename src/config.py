import os
from dotenv import load_dotenv

load_dotenv()

RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://127.0.0.1:8765")

# book_name from CSV -> filepath glob for Pathway filtering
BOOK_GLOBS = {
    "The Count of Monte Cristo": "**/The Count of Monte Cristo.txt",
    "In Search of the Castaways": "**/In search of the castaways.txt",
    "In search of the castaways": "**/In search of the castaways.txt",
}

NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
DEVICE = int(os.getenv("DEVICE", "-1"))  # -1 CPU, 0 GPU

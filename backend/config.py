import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "llm_model" / "llama-2-7b-chat.Q4_K_M.gguf"
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

LLM_CONTEXT_WINDOW = 2048
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.1

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jarvis-knowledge-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")


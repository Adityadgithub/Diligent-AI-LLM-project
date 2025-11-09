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


def _parse_environment(env: str | None):
    if not env:
        return None, None
    env = env.strip()
    lowered = env.lower()
    suffix_map = {
        "-aws": "aws",
        "-gcp": "gcp",
        "-azure": "azure",
    }
    for suffix, cloud in suffix_map.items():
        if lowered.endswith(suffix):
            region = env[: -len(suffix)]
            return cloud, region
    # default assumption when only region is provided (e.g. us-east-1)
    return "aws", env


PINECONE_CLOUD, PINECONE_REGION = _parse_environment(PINECONE_ENVIRONMENT)


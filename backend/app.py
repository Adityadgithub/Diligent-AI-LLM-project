import logging
from functools import lru_cache
from typing import List

import pinecone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.vectorstores import Pinecone as PineconeVectorStore

from config import (
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    MODEL_PATH,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ChatTurn(BaseModel):
    user: str = Field(..., description="End-user input")
    assistant: str = Field(..., description="Assistant reply")


class ChatRequest(BaseModel):
    message: str = Field(..., description="Latest user prompt")
    history: List[ChatTurn] = Field(default_factory=list, description="Ordered conversation history")


class ChatResponse(BaseModel):
    answer: str


app = FastAPI(title="LLM Knowledge Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_resources_exist():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}.")
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        raise RuntimeError(
            "PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set before starting the API."
        )


@lru_cache
def initialize_pinecone() -> None:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    indexes = pinecone.list_indexes()
    if PINECONE_INDEX_NAME not in indexes:
        raise RuntimeError(
            f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Run `python ingest.py` to build it."
        )


@lru_cache
def get_embeddings():
    logger.info("Loading sentence transformer embeddings: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_vector_store():
    embeddings = get_embeddings()
    initialize_pinecone()
    logger.info(
        "Connecting to Pinecone index '%s' (namespace: %s)",
        PINECONE_INDEX_NAME,
        PINECONE_NAMESPACE,
    )
    try:
        return PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=PINECONE_NAMESPACE,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to connect to Pinecone index: %s", exc)
        raise RuntimeError(
            "Unable to connect to Pinecone index. Verify your credentials and that ingestion has run."
        ) from exc


@lru_cache
def get_llm():
    logger.info("Initializing LlamaCpp with model %s", MODEL_PATH)
    return LlamaCpp(
        model_path=str(MODEL_PATH),
        n_ctx=LLM_CONTEXT_WINDOW,
        max_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=0.9,
        repeat_penalty=1.05,
        n_gpu_layers=-1,  # push all layers to the GPU
        n_batch=512,  # lower if you hit VRAM limits
        verbose=False,
    )


def format_history(history: List[ChatTurn]) -> str:
    if not history:
        return "No prior conversation."
    compiled = []
    for turn in history:
        compiled.append(f"User: {turn.user}\nAssistant: {turn.assistant}")
    return "\n".join(compiled)


PROMPT_TEMPLATE = """
[INST] <<SYS>>
You are Jarvis, a precise teaching assistant. You must answer strictly and only with information found in the supplied context snippets. Do not rely on outside knowledge or speculation. If the context does not contain the answer, reply exactly with: "I’m sorry, the knowledge base does not contain information about that."
Format your reply as clear paragraphs or bullet points that reference the context topic names when relevant.
<</SYS>>

Context:
{context}

Conversation so far:
{history}

Current user question:
{question}
[/INST]
"""

prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=PROMPT_TEMPLATE.strip(),
)


@app.on_event("startup")
async def startup_event():
    ensure_resources_exist()
    # Trigger lazy loading so startup errors surface early.
    get_embeddings()
    initialize_pinecone()
    get_vector_store()
    get_llm()
    logger.info("Application startup complete.")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    docs = retriever.get_relevant_documents(request.message)
    if not docs:
        raise HTTPException(status_code=404, detail="No relevant context found. Please ingest data first.")

    context = "\n\n".join(doc.page_content for doc in docs)
    history_text = format_history(request.history)

    rendered_prompt = prompt.format(context=context, history=history_text, question=request.message)
    llm = get_llm()
    answer = llm.invoke(rendered_prompt)

    return ChatResponse(answer=answer.strip())


import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import pinecone
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PineconeVectorStore

from config import (
    DATA_DIR,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_documents(data_dir: Path):
    documents = []
    for path in sorted(data_dir.glob("*.txt")):
        logging.info("Loading %s", path.name)
        loader = TextLoader(str(path), encoding="utf-8")
        documents.extend(loader.load())
    if not documents:
        raise FileNotFoundError(f"No .txt files found under {data_dir}.")
    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documents)


def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_pinecone_index(recreate: bool = False):
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        raise EnvironmentError(
            "PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set before ingesting data."
        )

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    existing_indexes = pinecone.list_indexes()
    if PINECONE_INDEX_NAME in existing_indexes and recreate:
        logging.info("Deleting existing Pinecone index '%s'...", PINECONE_INDEX_NAME)
        pinecone.delete_index(PINECONE_INDEX_NAME)
        while PINECONE_INDEX_NAME in pinecone.list_indexes():
            time.sleep(1)

    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        logging.info("Creating Pinecone index '%s'...", PINECONE_INDEX_NAME)
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            metric="cosine",
            dimension=EMBEDDING_DIMENSION,
            pods=1,
            pod_type="s1.x1",
        )
        while True:
            description = pinecone.describe_index(PINECONE_INDEX_NAME)
            status = description.get("status", {})
            if status.get("ready"):
                break
            logging.info("Waiting for Pinecone index to be ready...")
            time.sleep(2)
    else:
        logging.info("Using existing Pinecone index '%s'.", PINECONE_INDEX_NAME)


def upsert_chunks(chunks: Iterable, embeddings, namespace: str):
    logging.info(
        "Upserting %d chunks into Pinecone index '%s' (namespace: %s)...",
        len(chunks),
        PINECONE_INDEX_NAME,
        namespace,
    )
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
    )


def main(force: bool = False):
    ensure_pinecone_index(recreate=force)

    documents = load_documents(DATA_DIR)
    chunks = create_chunks(documents)
    embeddings = init_embeddings()

    if not force:
        index = pinecone.Index(PINECONE_INDEX_NAME)
        logging.info(
            "Clearing namespace '%s' before upserting new data.", PINECONE_NAMESPACE
        )
        index.delete(deleteAll=True, namespace=PINECONE_NAMESPACE)

    upsert_chunks(chunks, embeddings, namespace=PINECONE_NAMESPACE)
    logging.info(
        "Ingestion complete. Pinecone index '%s' now contains the latest data.",
        PINECONE_INDEX_NAME,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest domain documents into Pinecone.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the Pinecone index (destroys existing vectors).",
    )
    args = parser.parse_args()
    main(force=args.force)


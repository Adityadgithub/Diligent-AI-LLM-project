import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from config import (
    DATA_DIR,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
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


def init_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY must be set before ingesting data.")
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_pinecone_index(client: Pinecone, recreate: bool = False):
    index_list = client.list_indexes().names()
    if PINECONE_INDEX_NAME in index_list and recreate:
        logging.info("Deleting existing Pinecone index '%s'...", PINECONE_INDEX_NAME)
        client.delete_index(PINECONE_INDEX_NAME)
        while PINECONE_INDEX_NAME in client.list_indexes().names():
            time.sleep(0.5)

    if PINECONE_INDEX_NAME not in client.list_indexes().names():
        cloud = PINECONE_CLOUD or "gcp"
        region = PINECONE_REGION or "us-east1"
        logging.info(
            "Creating Pinecone serverless index '%s' (cloud=%s, region=%s)...",
            PINECONE_INDEX_NAME,
            cloud,
            region,
        )
        client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while True:
            description = client.describe_index(PINECONE_INDEX_NAME)
            status = description.get("status", {})
            if status.get("ready"):
                break
            logging.info("Waiting for Pinecone index to be ready...")
            time.sleep(2)
    else:
        logging.info("Using existing Pinecone index '%s'.", PINECONE_INDEX_NAME)


def upsert_chunks(client: Pinecone, chunks: Iterable, embeddings, namespace: str):
    index = client.Index(PINECONE_INDEX_NAME)
    chunk_list = list(chunks)
    logging.info(
        "Upserting %d chunks into Pinecone index '%s' (namespace: %s)...",
        len(chunk_list),
        PINECONE_INDEX_NAME,
        namespace,
    )
    batch_size = 64
    counter = 0
    for start in range(0, len(chunk_list), batch_size):
        batch = chunk_list[start : start + batch_size]
        texts = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(texts)
        payloads = []
        for vector, doc in zip(vectors, batch):
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["text"] = doc.page_content
            metadata.setdefault("chunk_id", f"{counter}")
            payloads.append(
                {
                    "id": f"{namespace}-{counter}",
                    "values": list(map(float, vector)),
                    "metadata": metadata,
                }
            )
            counter += 1
        index.upsert(vectors=payloads, namespace=namespace)


def clear_namespace(client: Pinecone, namespace: str):
    logging.info("Clearing namespace '%s' before upserting new data.", namespace)
    index = client.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as exc:
        logging.debug("Namespace deletion ignored: %s", exc)


def main(force: bool = False):
    client = init_client()
    ensure_pinecone_index(client, recreate=force)

    documents = load_documents(DATA_DIR)
    chunks = create_chunks(documents)
    embeddings = init_embeddings()

    clear_namespace(client, PINECONE_NAMESPACE)
    upsert_chunks(client, chunks, embeddings, namespace=PINECONE_NAMESPACE)
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


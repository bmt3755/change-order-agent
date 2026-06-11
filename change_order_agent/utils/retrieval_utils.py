from __future__ import annotations

import logging
import os
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
EMBEDDING_MODEL = "text-embedding-3-small"

# Collection names — shared constants used by agents and seeding scripts
CONTRACT_COLLECTION = "contract_corpus"
HISTORICAL_COLLECTION = "historical_cos"


def get_collection(collection_name: str) -> chromadb.Collection:
    """Return a persistent ChromaDB collection with OpenAI embeddings attached."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var="OPENAI_API_KEY",
        model_name=EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef,
    )


def retrieve(
    query: str,
    collection_name: str,
    top_k: int,
    where: dict[str, Any] | None = None,
) -> list[str]:
    """
    Embed query, search collection, return top_k document strings.
    Returns empty list on any failure — callers decide how to handle no results.
    """
    try:
        collection = get_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )
        documents: list[str] = results["documents"][0] if results["documents"] else []
        logger.info(
            "Retrieved %d results from '%s' for query: %.80s...",
            len(documents), collection_name, query,
        )
        return documents
    except Exception as exc:
        logger.error("Retrieval failed for collection '%s': %s", collection_name, exc)
        return []

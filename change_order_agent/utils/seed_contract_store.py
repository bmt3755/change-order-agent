"""
Run once at project setup to index the contract corpus.
Usage:
    python -m change_order_agent.utils.seed_contract_store \
        --file path/to/contract.txt \
        --org-id ORG123 \
        --project-id PROJ456 \
        --contract-version v1.0
"""
from __future__ import annotations

import argparse
import logging
import uuid

from .retrieval_utils import CONTRACT_COLLECTION, get_collection

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000    # characters per chunk
CHUNK_OVERLAP = 150  # overlap keeps context across chunk boundaries


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def seed_contract(
    file_path: str,
    org_id: str,
    project_id: str,
    contract_version: str,
) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = chunk_text(full_text)
    logger.info("Seeding contract: %d chunks from %s", len(chunks), file_path)

    collection = get_collection(CONTRACT_COLLECTION)

    # Remove any existing documents for this contract version to allow re-seeding
    try:
        collection.delete(
            where={
                "$and": [
                    {"org_id": {"$eq": org_id}},
                    {"project_id": {"$eq": project_id}},
                    {"contract_version": {"$eq": contract_version}},
                ]
            }
        )
        logger.info("Cleared existing chunks for contract version %s", contract_version)
    except Exception:
        pass  # collection may be empty on first seed

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "org_id": org_id,
            "project_id": project_id,
            "contract_version": contract_version,
            "doc_type": "contract",
            "chunk_index": i,
        }
        for i, _ in enumerate(chunks)
    ]

    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    logger.info("Seeded %d contract chunks into '%s'", len(chunks), CONTRACT_COLLECTION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--org-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--contract-version", required=True)
    args = parser.parse_args()
    seed_contract(args.file, args.org_id, args.project_id, args.contract_version)

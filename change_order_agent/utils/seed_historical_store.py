"""
Run once at project setup (and after each settled CO) to index historical change orders.
Input JSON format — list of objects:
[
  {
    "co_id": "CO-001",
    "work_type": "electrical",
    "description": "Additional conduit runs to new MRI room",
    "requested_amount": 45000,
    "settled_amount": 38000,
    "scope_ruling": "IN_SCOPE"
  },
  ...
]

Usage:
    python -m change_order_agent.utils.seed_historical_store \
        --file path/to/historical_cos.json \
        --org-id ORG123 \
        --project-id PROJ456
"""
from __future__ import annotations

import argparse
import json
import logging
import uuid

from .retrieval_utils import HISTORICAL_COLLECTION, get_collection

logger = logging.getLogger(__name__)


def _format_document(co: dict) -> str:
    """Format a historical CO as a plain text document for embedding."""
    return (
        f"Work type: {co.get('work_type', 'unknown')}\n"
        f"Description: {co.get('description', '')}\n"
        f"Amount requested: ${co.get('requested_amount', 0):,.0f}\n"
        f"Amount settled: ${co.get('settled_amount', 0):,.0f}\n"
        f"Scope ruling: {co.get('scope_ruling', 'unknown')}"
    )


def seed_historical(
    file_path: str,
    org_id: str,
    project_id: str,
) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        historical_cos: list[dict] = json.load(f)

    logger.info("Seeding %d historical COs from %s", len(historical_cos), file_path)

    collection = get_collection(HISTORICAL_COLLECTION)

    documents = [_format_document(co) for co in historical_cos]
    ids = [co.get("co_id", str(uuid.uuid4())) for co in historical_cos]
    metadatas = [
        {
            "org_id": org_id,
            "project_id": project_id,
            "doc_type": "historical_co",
            "work_type": co.get("work_type", "unknown"),
            "settled_amount": co.get("settled_amount", 0),
            "scope_ruling": co.get("scope_ruling", "unknown"),
        }
        for co in historical_cos
    ]

    # Use upsert so re-seeding after a CO is settled doesn't create duplicates
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("Upserted %d historical COs into '%s'", len(historical_cos), HISTORICAL_COLLECTION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--org-id", required=True)
    parser.add_argument("--project-id", required=True)
    args = parser.parse_args()
    seed_historical(args.file, args.org_id, args.project_id)

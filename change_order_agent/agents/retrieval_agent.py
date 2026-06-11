from __future__ import annotations

import logging
from datetime import datetime, timezone

from langsmith import traceable

from ..state.change_order_state import ChangeOrderState, PipelineStatus, RetrievalOutput
from ..utils.retrieval_utils import CONTRACT_COLLECTION, retrieve

logger = logging.getLogger(__name__)

TOP_K_CONTRACT = 6  # more sections needed for scope comparison than for cost estimation


def _build_query(state: ChangeOrderState) -> str:
    """Build a targeted retrieval query from extracted key facts."""
    parts = []
    if state.extraction.work_type:
        parts.append(state.extraction.work_type)
    if state.extraction.description:
        parts.append(state.extraction.description)
    return " ".join(parts) if parts else (state.input.redacted_document or "")


@traceable(name="retrieval_agent")
def run_retrieval_agent(state: ChangeOrderState) -> dict:
    """Task 2: Retrieve contract scope and relevant spec sections."""
    co_id = state.input.co_id
    logger.info("CO %s: retrieval_agent starting", co_id)

    query = _build_query(state)

    # Metadata filter — org + project + contract version (Step 6 consistency requirement)
    where_filter = {
        "$and": [
            {"org_id": {"$eq": state.input.org_id}},
            {"project_id": {"$eq": state.input.project_id}},
            {"contract_version": {"$eq": state.input.contract_version}},
        ]
    }

    sections = retrieve(
        query=query,
        collection_name=CONTRACT_COLLECTION,
        top_k=TOP_K_CONTRACT,
        where=where_filter,
    )

    if not sections:
        logger.warning(
            "CO %s: no contract sections retrieved — scope analysis will have no contract context",
            co_id,
        )
        # Parallel agents never write to "pipeline" — error goes to retrieval.error
        return {
            "retrieval": RetrievalOutput(
                retrieved_at=datetime.now(timezone.utc),
                error=f"CO {co_id}: no contract sections found for version {state.input.contract_version}",
            ),
        }

    logger.info("CO %s: retrieved %d contract sections", co_id, len(sections))

    return {
        "retrieval": RetrievalOutput(
            contract_sections=sections,
            retrieved_at=datetime.now(timezone.utc),
        ),
    }

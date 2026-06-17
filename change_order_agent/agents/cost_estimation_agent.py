from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel, Field

from ..state.change_order_state import (
    ChangeOrderState,
    CostEstimationOutput,
)
from ..utils.retrieval_utils import HISTORICAL_COLLECTION, retrieve

logger = logging.getLogger(__name__)

client = OpenAI()
MODEL = "gpt-4o-mini"
TOP_K_HISTORICAL = 4  # fewer comparables needed than contract sections


class _CostEstimateResult(BaseModel):
    estimated_cost_low: float = Field(ge=0)
    estimated_cost_high: float = Field(ge=0)
    reasoning: str  # brief explanation — included in audit trail


_ESTIMATION_PROMPT = """\
You are a construction cost analyst. Based on these similar historical change orders, \
estimate a fair cost range (low and high) for the new change order described below.

Historical comparables:
{comparables}

New change order:
Work type: {work_type}
Description: {description}
Amount requested by subcontractor: {requested_amount}

Return estimated_cost_low, estimated_cost_high (dollar amounts), and one sentence of reasoning."""


def _build_query(state: ChangeOrderState) -> str:
    parts = []
    if state.extraction.work_type:
        parts.append(state.extraction.work_type)
    if state.extraction.description:
        parts.append(state.extraction.description)
    if state.extraction.dollar_amount_requested is not None:
        parts.append(f"${state.extraction.dollar_amount_requested:,.0f}")
    return " ".join(parts) if parts else (state.input.redacted_document or "")


@traceable(name="cost_estimation_agent")
def run_cost_estimation_agent(state: ChangeOrderState) -> dict:
    """Task 4: Estimate cost impact using historical change order data."""
    co_id = state.input.co_id
    logger.info("CO %s: cost_estimation_agent starting", co_id)

    query = _build_query(state)

    where_filter = {
        "$and": [
            {"org_id": {"$eq": state.input.org_id}},
            {"project_id": {"$eq": state.input.project_id}},
        ]
    }

    comparables = retrieve(
        query=query,
        collection_name=HISTORICAL_COLLECTION,
        top_k=TOP_K_HISTORICAL,
        where=where_filter,
    )

    # Parallel agents never write to "pipeline" — errors go to cost_estimation.error
    if not comparables:
        logger.warning("CO %s: no historical comparables — cost estimation unavailable", co_id)
        return {
            "cost_estimation": CostEstimationOutput(
                estimated_at=datetime.now(timezone.utc),
                error=f"CO {co_id}: no historical data — a human must provide cost benchmark manually",
            ),
        }

    result = _estimate_cost(comparables, state, co_id)

    if result is None:
        logger.error("CO %s: LLM cost estimation failed", co_id)
        return {
            "cost_estimation": CostEstimationOutput(
                historical_comparables=comparables,
                estimated_at=datetime.now(timezone.utc),
                error=f"CO {co_id}: cost estimation LLM call failed",
            ),
        }

    logger.info(
        "CO %s: cost estimation complete — range %s",
        co_id,
        f"${result.estimated_cost_low:,.0f}–${result.estimated_cost_high:,.0f}",
    )

    return {
        "cost_estimation": CostEstimationOutput(
            estimated_cost_low=result.estimated_cost_low,
            estimated_cost_high=result.estimated_cost_high,
            historical_comparables=comparables,
            estimated_at=datetime.now(timezone.utc),
        ),
    }


def _estimate_cost(
    comparables: list[str],
    state: ChangeOrderState,
    co_id: str,
) -> Optional[_CostEstimateResult]:
    try:
        response = client.beta.chat.completions.parse(
            model=MODEL,
            temperature=0,  # deterministic — legal consistency requirement (Step 6)
            messages=[
                {"role": "system", "content": "You are a construction cost analyst. Return only structured data."},
                {
                    "role": "user",
                    "content": _ESTIMATION_PROMPT.format(
                        comparables="\n---\n".join(comparables),
                        work_type=state.extraction.work_type or "unknown",
                        description=state.extraction.description or "not provided",
                        requested_amount=(
                            f"${state.extraction.dollar_amount_requested:,.0f}"
                            if state.extraction.dollar_amount_requested is not None
                            else "not stated"
                        ),
                    ),
                },
            ],
            response_format=_CostEstimateResult,
        )
        return response.choices[0].message.parsed
    except Exception as exc:
        logger.error("CO %s: cost estimation LLM call failed: %s", co_id, exc)
        return None

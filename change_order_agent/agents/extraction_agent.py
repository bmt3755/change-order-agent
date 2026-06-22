from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from langsmith import traceable
from pydantic import BaseModel, Field

from ..state.change_order_state import (
    ChangeOrderState,
    ExtractionOutput,
    PipelineStatus,
)
from ..utils.llm import get_client

logger = logging.getLogger(__name__)

client = get_client()  # bounded timeout + retries (utils/llm.py)

MODEL = "gpt-4o-mini"
MAX_PRIMARY_ATTEMPTS = 2  # one retry before falling back


# ---------------------------------------------------------------------------
# LLM response models — used only for parsing; stricter state model lives in state/
# ---------------------------------------------------------------------------

class _PrimaryResult(BaseModel):
    work_type: str
    subcontractor_name: str
    dollar_amount_requested: Optional[float] = Field(default=None, ge=0)
    description: str


class _FallbackResult(BaseModel):
    work_type: Optional[str] = None
    dollar_amount_requested: Optional[float] = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PRIMARY_PROMPT = """\
You are a construction document analyst. Read the change order below and extract:

1. work_type — type of construction work (e.g. electrical, structural, mechanical, plumbing)
2. subcontractor_name — name of the subcontractor submitting this change order
3. dollar_amount_requested — dollar amount requested (number only, no $ sign). Null if not stated.
4. description — one plain-English sentence describing what work is being requested

Change order:
{document}"""

_FALLBACK_PROMPT = """\
Read the change order below. Extract only two fields:
1. work_type — type of construction work (electrical, structural, mechanical, plumbing, or other). Null if unclear.
2. dollar_amount_requested — dollar amount requested (number only). Null if not stated.

Change order:
{document}"""


# ---------------------------------------------------------------------------
# Agent node — called by LangGraph orchestrator
# ---------------------------------------------------------------------------

@traceable(name="extraction_agent")
def run_extraction_agent(state: ChangeOrderState) -> dict:
    """Task 1: Read the change order and extract key facts."""
    co_id = state.input.co_id
    document = state.input.redacted_document

    if document is None:
        logger.error("CO %s: redacted_document is None — cannot extract", co_id)
        return _failed(state, "redacted_document is None")

    logger.info("CO %s: extraction_agent starting", co_id)

    # --- Primary path ---
    result = _try_primary(document, co_id)

    # --- Fallback path ---
    if result is None:
        logger.warning("CO %s: primary extraction failed — trying fallback", co_id)
        result = _try_fallback(document, co_id)

    # --- Flag-for-review path ---
    if result is None:
        logger.error("CO %s: both extraction paths failed — flagging for review", co_id)
        return _flag_for_review(state)

    logger.info(
        "CO %s: extraction complete — work_type=%s amount=%s flagged=%s",
        co_id, result.work_type, result.dollar_amount_requested, result.flagged_missing_amount,
    )

    return {
        "extraction": result,
        "pipeline": state.pipeline.model_copy(update={"current_node": "extraction_agent"}),
    }


# ---------------------------------------------------------------------------
# Primary extraction — full fields, one retry allowed
# ---------------------------------------------------------------------------

def _try_primary(document: str, co_id: str) -> Optional[ExtractionOutput]:
    for attempt in range(1, MAX_PRIMARY_ATTEMPTS + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                temperature=0,  # deterministic — required for legal consistency (Step 6)
                messages=[
                    {"role": "system", "content": "Return only structured data. No commentary."},
                    {"role": "user", "content": _PRIMARY_PROMPT.format(document=document)},
                ],
                response_format=_PrimaryResult,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.warning("CO %s: primary attempt %d — model returned None", co_id, attempt)
                continue

            return ExtractionOutput(
                work_type=parsed.work_type,
                subcontractor_name=parsed.subcontractor_name,
                dollar_amount_requested=parsed.dollar_amount_requested,
                description=parsed.description,
                flagged_missing_amount=parsed.dollar_amount_requested is None,
                extracted_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.warning("CO %s: primary attempt %d failed: %s", co_id, attempt, exc)

    return None


# ---------------------------------------------------------------------------
# Fallback extraction — critical fields only, no retry
# ---------------------------------------------------------------------------

def _try_fallback(document: str, co_id: str) -> Optional[ExtractionOutput]:
    try:
        response = client.beta.chat.completions.parse(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return only the fields requested. Null for anything not found."},
                {"role": "user", "content": _FALLBACK_PROMPT.format(document=document)},
            ],
            response_format=_FallbackResult,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return None

        logger.info("CO %s: fallback extraction succeeded", co_id)
        return ExtractionOutput(
            work_type=parsed.work_type,
            dollar_amount_requested=parsed.dollar_amount_requested,
            flagged_missing_amount=parsed.dollar_amount_requested is None,
            extracted_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        logger.error("CO %s: fallback extraction failed: %s", co_id, exc)
        return None


# ---------------------------------------------------------------------------
# State update helpers
# ---------------------------------------------------------------------------

def _flag_for_review(state: ChangeOrderState) -> dict:
    return {
        "extraction": ExtractionOutput(
            flagged_missing_amount=True,
            extracted_at=datetime.now(timezone.utc),
        ),
        "pipeline": state.pipeline.model_copy(update={
            "status": PipelineStatus.AWAITING_REVIEW,
            "current_node": "extraction_agent",
            "error_message": f"CO {state.input.co_id}: extraction failed on both paths",
        }),
    }


def _failed(state: ChangeOrderState, reason: str) -> dict:
    return {
        "extraction": ExtractionOutput(flagged_missing_amount=True),
        "pipeline": state.pipeline.model_copy(update={
            "status": PipelineStatus.FAILED,
            "current_node": "extraction_agent",
            "error_message": reason,
        }),
    }

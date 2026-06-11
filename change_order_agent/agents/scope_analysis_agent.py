from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel, Field

from ..state.change_order_state import (
    ChangeOrderState,
    PipelineStatus,
    ScopeAnalysisOutput,
    ScopeRuling,
)

logger = logging.getLogger(__name__)

client = OpenAI()
MODEL = "gpt-4o-mini"
MAX_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# LLM response model — citation is required; ruling without a clause is not accepted
# ---------------------------------------------------------------------------

class _ScopeAnalysisResult(BaseModel):
    scope_ruling: ScopeRuling
    confidence_score: float = Field(ge=0.0, le=1.0)
    contract_clause_cited: str  # exact clause text from the contract — required for audit trail
    reasoning: str              # one sentence explaining the ruling


_SCOPE_PROMPT = """\
You are a construction contract analyst reviewing a change order for a hospital project.

CRITICAL CONTEXT — READ BEFORE ANALYZING:
All change orders are requests for work beyond the original drawings. "In scope" does NOT \
mean "in the original drawings." It means the TYPE of work is PERMITTED under the contract \
and eligible for standard change order approval.

Use these definitions strictly:
- IN_SCOPE: The contract does NOT explicitly prohibit this type of work. Standard change \
  order approval applies based on dollar threshold.
- OUT_OF_SCOPE: The contract contains an explicit clause that NAMES or DESCRIBES this \
  specific type of work as excluded or prohibited.
- AMBIGUOUS: A relevant clause exists but it is genuinely unclear whether it covers \
  this specific request.

NARROW EXCLUSION RULE: An exclusion clause applies ONLY to the exact work it names. \
A clause excluding "electrical work in specialized medical equipment rooms" excludes only \
work inside those specific rooms — it does NOT exclude all electrical work on the project. \
Do not expand an exclusion beyond what the contract text explicitly states.

CONTRACT SECTIONS:
{contract_sections}

CHANGE ORDER:
Work type: {work_type}
Subcontractor: {subcontractor_name}
Amount requested: {dollar_amount}
Description: {description}

Steps:
1. Identify the most relevant contract clause.
2. Ask: Does this clause EXPLICITLY name this specific type of work as excluded?
3. Apply the narrow exclusion rule — do not expand an exclusion beyond its literal text.
4. Rule IN_SCOPE, OUT_OF_SCOPE, or AMBIGUOUS.
5. Score confidence (0.75–1.0 = clear, 0.45–0.74 = requires interpretation, \
   0.0–0.44 = genuinely unclear).

Return:
- scope_ruling
- confidence_score
- contract_clause_cited: exact clause text from the contract above
- reasoning: one sentence citing the specific work requested and whether the clause \
  explicitly names it as excluded or permitted"""


# ---------------------------------------------------------------------------
# Agent node — called by LangGraph orchestrator
# ---------------------------------------------------------------------------

@traceable(name="scope_analysis_agent")
def run_scope_analysis_agent(state: ChangeOrderState) -> dict:
    """Task 3: Compare scope — in or out. Depends on state.extraction and state.retrieval."""
    co_id = state.input.co_id
    logger.info("CO %s: scope_analysis_agent starting", co_id)

    # Check for errors from parallel agents (retrieval and cost_estimation)
    if state.retrieval.error:
        logger.warning("CO %s: retrieval error noted — %s", co_id, state.retrieval.error)
    if state.cost_estimation.error:
        logger.warning("CO %s: cost estimation error noted — %s", co_id, state.cost_estimation.error)

    if not state.retrieval.contract_sections:
        logger.error("CO %s: no contract sections — cannot determine scope", co_id)
        return _flag_for_david(
            state,
            f"CO {co_id}: scope analysis skipped — no contract sections in state.retrieval",
        )

    result = _try_scope_analysis(state, co_id)

    if result is None:
        logger.error(
            "CO %s: scope analysis failed after %d attempts — flagging for David",
            co_id, MAX_ATTEMPTS,
        )
        return _flag_for_david(
            state,
            f"CO {co_id}: scope analysis LLM call failed on all attempts",
        )

    tier = (
        "HIGH" if result.confidence_score >= 0.75
        else "MEDIUM" if result.confidence_score >= 0.45
        else "LOW"
    )
    logger.info(
        "CO %s: scope analysis complete — ruling=%s confidence=%.2f tier=%s",
        co_id, result.scope_ruling.value, result.confidence_score, tier,
    )

    return {
        "scope_analysis": ScopeAnalysisOutput(
            scope_ruling=result.scope_ruling,
            confidence_score=result.confidence_score,
            contract_clause_cited=result.contract_clause_cited,
            reasoning=result.reasoning,
            analyzed_at=datetime.now(timezone.utc),
        ),
        "pipeline": state.pipeline.model_copy(update={"current_node": "scope_analysis_agent"}),
    }


# ---------------------------------------------------------------------------
# LLM call — one retry before giving up
# ---------------------------------------------------------------------------

def _try_scope_analysis(
    state: ChangeOrderState, co_id: str
) -> Optional[_ScopeAnalysisResult]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                temperature=0,  # deterministic — legal consistency requirement (Step 6)
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a construction contract analyst. "
                            "Return only structured data. "
                            "Never guess — use AMBIGUOUS with a low confidence score "
                            "when the contract language is unclear."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _SCOPE_PROMPT.format(
                            contract_sections="\n---\n".join(
                                state.retrieval.contract_sections
                            ),
                            work_type=state.extraction.work_type or "unknown",
                            subcontractor_name=state.extraction.subcontractor_name or "unknown",
                            dollar_amount=(
                                f"${state.extraction.dollar_amount_requested:,.0f}"
                                if state.extraction.dollar_amount_requested is not None
                                else "not stated"
                            ),
                            description=state.extraction.description or "not provided",
                        ),
                    },
                ],
                response_format=_ScopeAnalysisResult,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.warning(
                    "CO %s: scope analysis attempt %d returned None", co_id, attempt
                )
                continue
            return parsed
        except Exception as exc:
            logger.warning(
                "CO %s: scope analysis attempt %d failed: %s", co_id, attempt, exc
            )

    return None


# ---------------------------------------------------------------------------
# State update helpers
# ---------------------------------------------------------------------------

def _flag_for_david(state: ChangeOrderState, reason: str) -> dict:
    return {
        "scope_analysis": ScopeAnalysisOutput(
            analyzed_at=datetime.now(timezone.utc),
        ),
        "pipeline": state.pipeline.model_copy(update={
            "status": PipelineStatus.AWAITING_DAVID,
            "current_node": "scope_analysis_agent",
            "error_message": reason,
        }),
    }

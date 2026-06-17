from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel

from ..state.change_order_state import (
    ApprovalStage,
    ApproverLevel,
    ChangeOrderState,
    ConfidenceTier,
    PipelineStatus,
    RiskScore,
    AssemblyOutput,
    ScopeRuling,
)

logger = logging.getLogger(__name__)

client = OpenAI()
MODEL = "gpt-4o-mini"
MAX_DRAFT_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Risk score rules — evaluated in priority order, first match wins
# Default to HIGH when inputs are missing (when in doubt, escalate)
# ---------------------------------------------------------------------------

HIGH_COST_THRESHOLD = 500_000
MEDIUM_COST_THRESHOLD = 100_000


def _determine_risk_score(state: ChangeOrderState) -> RiskScore:
    sa = state.scope_analysis
    ce = state.cost_estimation

    scope_ruling = sa.scope_ruling
    confidence_tier = sa.confidence_tier
    cost_high = ce.estimated_cost_high

    # HIGH conditions — any one triggers HIGH
    if (
        scope_ruling == ScopeRuling.OUT_OF_SCOPE
        or confidence_tier == ConfidenceTier.LOW
        or (cost_high is not None and cost_high > HIGH_COST_THRESHOLD)
        or scope_ruling is None  # missing ruling — default to HIGH
    ):
        return RiskScore.HIGH

    # MEDIUM conditions — any one triggers MEDIUM
    if (
        scope_ruling == ScopeRuling.AMBIGUOUS
        or confidence_tier == ConfidenceTier.MEDIUM
        or (cost_high is not None and cost_high > MEDIUM_COST_THRESHOLD)
    ):
        return RiskScore.MEDIUM

    return RiskScore.LOW


# ---------------------------------------------------------------------------
# Approval stage — derived from routing output
# ---------------------------------------------------------------------------

_APPROVER_TO_STAGE: dict[ApproverLevel, ApprovalStage] = {
    ApproverLevel.SITE_SUPER:       ApprovalStage.WAITING_ON_ENGINEER,
    ApproverLevel.PROJECT_MANAGER:  ApprovalStage.WAITING_ON_ENGINEER,
    ApproverLevel.OWNERS_REP:       ApprovalStage.WAITING_ON_OWNERS_REP,
    ApproverLevel.OWNER:            ApprovalStage.WAITING_ON_OWNER,
}


def _determine_approval_stage(state: ChangeOrderState) -> ApprovalStage:
    if not state.routing.routing_executed:
        return ApprovalStage.PENDING

    # Out of scope always escalated regardless of routing level
    if state.scope_analysis.scope_ruling == ScopeRuling.OUT_OF_SCOPE:
        return ApprovalStage.ESCALATED

    if state.routing.approver_level is None:
        return ApprovalStage.PENDING

    return _APPROVER_TO_STAGE.get(state.routing.approver_level, ApprovalStage.PENDING)


# ---------------------------------------------------------------------------
# Escalation draft — LLM generates professional email, the reviewer reviews before sending
# ---------------------------------------------------------------------------

class _DraftResult(BaseModel):
    subject: str
    body: str


_DRAFT_PROMPT = """\
Write a professional email from a project manager to the approver listed below regarding a construction change order.

IMPORTANT: Use ONLY the figures and facts provided. Do not rephrase, round, or invent any dollar amounts.

FACTS:
CO ID: {co_id}
Project: {project_id}
Subcontractor: {subcontractor}
Work type: {work_type}
Description: {description}
Amount requested: {amount_requested}
Estimated fair range: {cost_range}
Scope ruling: {scope_ruling}
Confidence: {confidence}
Route to: {approver_level} — {department}
Risk level: {risk_score}

Write:
- subject: a concise email subject line
- body: 3–4 short paragraphs. Paragraph 1: what is being requested. \
Paragraph 2: scope ruling and confidence. Paragraph 3: cost context. \
Paragraph 4: what action is needed from the approver.

Do not add a greeting or sign-off — the reviewer will personalise those before sending."""

_TEMPLATE_DRAFT = """\
Subject: Change Order {co_id} — {work_type} — Action Required

This is an automatically generated draft. Please review and personalise before sending.

Change Order {co_id} has been submitted by {subcontractor} for {work_type} work on project {project_id}.

Description: {description}

Scope ruling: {scope_ruling} (confidence: {confidence})
Amount requested: {amount_requested} | Estimated fair range: {cost_range}

This change order has been routed to {approver_level} ({department}) for approval.
Risk level: {risk_score}

Please review the attached supporting documents and confirm your approval or escalation decision."""


def _build_draft_context(state: ChangeOrderState, risk_score: RiskScore) -> dict:
    ex = state.extraction
    sa = state.scope_analysis
    ce = state.cost_estimation
    ro = state.routing

    return {
        "co_id": state.input.co_id,
        "project_id": state.input.project_id,
        "subcontractor": ex.subcontractor_name or "unknown",
        "work_type": ex.work_type or "unknown",
        "description": ex.description or "not provided",
        "amount_requested": (
            f"${ex.dollar_amount_requested:,.0f}"
            if ex.dollar_amount_requested is not None else "not stated"
        ),
        "cost_range": (
            f"${ce.estimated_cost_low:,.0f}–${ce.estimated_cost_high:,.0f}"
            if ce.estimated_cost_low is not None and ce.estimated_cost_high is not None
            else "not available"
        ),
        "scope_ruling": sa.scope_ruling.value if sa.scope_ruling else "pending",
        "confidence": (
            f"{sa.confidence_score:.0%} ({sa.confidence_tier.value})"
            if sa.confidence_score is not None and sa.confidence_tier is not None
            else "unknown"
        ),
        "approver_level": ro.approver_level.value if ro.approver_level else "unknown",
        "department": ro.department or "unknown",
        "risk_score": risk_score.value,
    }


def _try_llm_draft(context: dict, co_id: str) -> Optional[str]:
    for attempt in range(1, MAX_DRAFT_ATTEMPTS + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                temperature=0,  # minimise hallucination risk on financial figures
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional construction project manager. Use only the facts provided. Never invent figures.",
                    },
                    {"role": "user", "content": _DRAFT_PROMPT.format(**context)},
                ],
                response_format=_DraftResult,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.warning("CO %s: draft attempt %d returned None", co_id, attempt)
                continue
            return f"Subject: {parsed.subject}\n\n{parsed.body}\n\n[DRAFT — requires review and approval before sending]"
        except Exception as exc:
            logger.warning("CO %s: draft attempt %d failed: %s", co_id, attempt, exc)
    return None


# ---------------------------------------------------------------------------
# Report assembly — the reviewer's full dashboard view
# ---------------------------------------------------------------------------

def _assemble_report(
    state: ChangeOrderState,
    approval_stage: ApprovalStage,
    risk_score: RiskScore,
) -> str:
    ex = state.extraction
    sa = state.scope_analysis
    ce = state.cost_estimation
    ro = state.routing

    lines = [
        "CHANGE ORDER STATUS REPORT",
        "=" * 52,
        f"CO ID:             {state.input.co_id}",
        f"Project:           {state.input.project_id}",
        f"Subcontractor:     {ex.subcontractor_name or 'unknown'}",
        f"Work type:         {ex.work_type or 'unknown'}",
        "",
        "STATUS",
        "-" * 52,
        f"Approval stage:    {approval_stage.value}",
        f"Risk score:        {risk_score.value}",
        f"Routed to:         {ro.approver_level.value if ro.approver_level else 'pending'} — {ro.department or 'unknown'}",
        "",
        "SCOPE",
        "-" * 52,
        f"Ruling:            {sa.scope_ruling.value if sa.scope_ruling else 'pending'}",
        f"Confidence:        {f'{sa.confidence_score:.0%} ({sa.confidence_tier.value})' if sa.confidence_score is not None and sa.confidence_tier else 'unknown'}",
        f"Clause cited:      {sa.contract_clause_cited or 'none'}",
        f"Reasoning:         {sa.reasoning or 'none'}",
        "",
        "COST",
        "-" * 52,
        f"Requested:         {'${:,.0f}'.format(ex.dollar_amount_requested) if ex.dollar_amount_requested else 'not stated'}",
        f"Estimated range:   {'${:,.0f}–${:,.0f}'.format(ce.estimated_cost_low, ce.estimated_cost_high) if ce.estimated_cost_low and ce.estimated_cost_high else 'not available'}",
        "",
        f"Generated:         {datetime.now(timezone.utc).isoformat()}",
        f"Contract version:  {state.input.contract_version}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent node — called by LangGraph orchestrator
# ---------------------------------------------------------------------------

@traceable(name="output_assembly_agent")
def run_output_assembly_agent(state: ChangeOrderState) -> dict:
    """Task 8: Assemble the final report and escalation draft for review."""
    co_id = state.input.co_id
    logger.info("CO %s: output_assembly_agent starting", co_id)

    # Deterministic risk score — always runs regardless of upstream state
    risk_score = _determine_risk_score(state)
    approval_stage = _determine_approval_stage(state)

    logger.info(
        "CO %s: risk_score=%s approval_stage=%s",
        co_id, risk_score.value, approval_stage.value,
    )

    # Parallel agents never write to "pipeline" — errors go to assembly.error
    try:
        full_report = _assemble_report(state, approval_stage, risk_score)
    except Exception as exc:
        logger.error("CO %s: report assembly failed: %s", co_id, exc)
        return {
            "assembly": AssemblyOutput(
                error=f"CO {co_id}: report assembly failed — {exc}",
            ),
        }

    # LLM escalation draft — falls back to template if LLM fails
    context = _build_draft_context(state, risk_score)
    draft = _try_llm_draft(context, co_id)

    if draft is None:
        logger.warning("CO %s: LLM draft failed — using template fallback", co_id)
        draft = (
            _TEMPLATE_DRAFT.format(**context)
            + "\n\n[DRAFT — auto-generated fallback — requires review and approval before sending]"
        )

    logger.info("CO %s: output assembly complete — report and draft ready for review", co_id)

    return {
        "assembly": AssemblyOutput(
            approval_stage=approval_stage,
            risk_score=risk_score,
            full_report=full_report,
            escalation_draft=draft,
            report_delivered=True,
            delivered_at=datetime.now(timezone.utc),
        ),
    }

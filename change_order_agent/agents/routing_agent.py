from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from langsmith import traceable

from ..state.change_order_state import (
    ApproverLevel,
    ChangeOrderState,
    PipelineStatus,
    RoutingOutput,
    ScopeRuling,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing configuration — update here when GC changes approval thresholds
# ---------------------------------------------------------------------------

# (upper_bound_exclusive, approver_level) — evaluated in order, first match wins
DOLLAR_THRESHOLDS: list[tuple[float, ApproverLevel]] = [
    (25_000,  ApproverLevel.SITE_SUPER),
    (100_000, ApproverLevel.PROJECT_MANAGER),
    (500_000, ApproverLevel.OWNERS_REP),
]
OWNER_LEVEL = ApproverLevel.OWNER  # > $500K or any OUT_OF_SCOPE ruling

WORK_TYPE_TO_DEPARTMENT: dict[str, str] = {
    "electrical": "MEP Engineer",
    "mechanical": "MEP Engineer",
    "plumbing":   "MEP Engineer",
    "structural": "Structural Engineer",
    "civil":      "Civil Engineer",
    "site work":  "Civil Engineer",
}
DEFAULT_DEPARTMENT = "Project Manager"  # used when work type is unknown or not in the table


# ---------------------------------------------------------------------------
# Routing logic — pure Python, no LLM, fully deterministic (Step 6)
# ---------------------------------------------------------------------------

def _determine_approver_level(
    dollar_amount: float,
    scope_ruling: ScopeRuling,
) -> ApproverLevel:
    # Out of scope always routes to owner regardless of dollar amount
    if scope_ruling == ScopeRuling.OUT_OF_SCOPE:
        return OWNER_LEVEL
    for threshold, level in DOLLAR_THRESHOLDS:
        if dollar_amount < threshold:
            return level
    return OWNER_LEVEL


def _determine_department(work_type: Optional[str]) -> str:
    if not work_type:
        logger.warning("Work type is unknown — defaulting department to Project Manager")
        return DEFAULT_DEPARTMENT
    return WORK_TYPE_TO_DEPARTMENT.get(work_type.lower().strip(), DEFAULT_DEPARTMENT)


def _resolve_dollar_amount(state: ChangeOrderState) -> Optional[float]:
    """Use requested amount if available; fall back to cost estimate high end."""
    if state.extraction.dollar_amount_requested is not None:
        return state.extraction.dollar_amount_requested
    if state.cost_estimation.estimated_cost_high is not None:
        logger.warning(
            "CO %s: dollar amount missing — using cost estimate high end (%s) for routing",
            state.input.co_id,
            f"${state.cost_estimation.estimated_cost_high:,.0f}",
        )
        return state.cost_estimation.estimated_cost_high
    return None


# ---------------------------------------------------------------------------
# Package assembly — supporting documents attached for the approver
# ---------------------------------------------------------------------------

def _assemble_package(
    state: ChangeOrderState,
    approver_level: ApproverLevel,
    department: str,
) -> str:
    sa = state.scope_analysis
    ce = state.cost_estimation
    ex = state.extraction

    amount_str = (
        f"${ex.dollar_amount_requested:,.0f}"
        if ex.dollar_amount_requested is not None
        else "not stated"
    )
    cost_range = (
        f"${ce.estimated_cost_low:,.0f}–${ce.estimated_cost_high:,.0f}"
        if ce.estimated_cost_low is not None and ce.estimated_cost_high is not None
        else "not available"
    )
    confidence_str = (
        f"{sa.confidence_score:.0%} ({sa.confidence_tier.value})"
        if sa.confidence_score is not None and sa.confidence_tier is not None
        else "unknown"
    )

    lines = [
        "CHANGE ORDER ROUTING PACKAGE",
        "=" * 52,
        f"CO ID:             {state.input.co_id}",
        f"Project:           {state.input.project_id}",
        f"Subcontractor:     {ex.subcontractor_name or 'unknown'}",
        f"Work type:         {ex.work_type or 'unknown'}",
        f"Department:        {department}",
        f"Amount requested:  {amount_str}",
        f"Estimated range:   {cost_range}",
        f"Description:       {ex.description or 'not provided'}",
        "",
        "SCOPE ANALYSIS",
        "-" * 52,
        f"Ruling:            {sa.scope_ruling.value if sa.scope_ruling else 'pending'}",
        f"Confidence:        {confidence_str}",
        f"Clause cited:      {sa.contract_clause_cited or 'none'}",
        f"Reasoning:         {sa.reasoning or 'none'}",
        "",
        "ROUTING DECISION",
        "-" * 52,
        # approver_name stores the role — actual contact lookup plugs in here (project management integration)
        f"Route to:          {approver_level.value}",
        f"Contract version:  {state.input.contract_version}",
        f"Cache version:     {state.input.cache_version}",
        f"Routed at:         {datetime.now(timezone.utc).isoformat()}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent node — called by LangGraph orchestrator
# ---------------------------------------------------------------------------

@traceable(name="routing_agent")
def run_routing_agent(state: ChangeOrderState) -> dict:
    """Tasks 5 and 6: Determine approval routing and dispatch with supporting documents."""
    co_id = state.input.co_id
    logger.info("CO %s: routing_agent starting", co_id)

    if state.scope_analysis.scope_ruling is None:
        logger.error("CO %s: scope ruling is None — cannot route", co_id)
        return _flag_for_review(
            state, f"CO {co_id}: routing skipped — no scope ruling available"
        )

    dollar_amount = _resolve_dollar_amount(state)

    if dollar_amount is None:
        logger.error("CO %s: no dollar amount or cost estimate — cannot determine approval level", co_id)
        return _flag_for_review(
            state,
            f"CO {co_id}: routing skipped — no dollar amount or cost estimate available",
        )

    # Task 5 — routing decision (deterministic rules table)
    approver_level = _determine_approver_level(dollar_amount, state.scope_analysis.scope_ruling)
    department = _determine_department(state.extraction.work_type)

    logger.info(
        "CO %s: routing decision — approver=%s department=%s dollar_basis=%s",
        co_id, approver_level.value, department, f"${dollar_amount:,.0f}",
    )

    # Task 6 — assemble package and dispatch
    try:
        package = _assemble_package(state, approver_level, department)
        logger.info("CO %s: routing package assembled\n%s", co_id, package)
        # TODO: deliver package via project management API or email — plug in here
        routing_executed = True
    except Exception as exc:
        logger.error("CO %s: package assembly failed: %s", co_id, exc)
        return _flag_for_review(
            state, f"CO {co_id}: routing package assembly failed — {exc}"
        )

    return {
        "routing": RoutingOutput(
            approver_name=approver_level.value,
            approver_level=approver_level,
            department=department,
            routing_executed=routing_executed,
            routed_at=datetime.now(timezone.utc),
        ),
        "pipeline": state.pipeline.model_copy(
            update={"current_node": "routing_agent"}
        ),
    }


# ---------------------------------------------------------------------------
# State update helper
# ---------------------------------------------------------------------------

def _flag_for_review(state: ChangeOrderState, reason: str) -> dict:
    return {
        "routing": RoutingOutput(),
        "pipeline": state.pipeline.model_copy(update={
            "status": PipelineStatus.AWAITING_REVIEW,
            "current_node": "routing_agent",
            "error_message": reason,
        }),
    }

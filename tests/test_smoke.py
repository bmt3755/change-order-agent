"""
Smoke tests — no OpenAI API calls, no vector store, no SQLite writes.
Verifies: imports resolve, state schema validates, business logic is correct,
graph compiles. Run these before spending any API tokens.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Minimal valid input reused across tests
# ---------------------------------------------------------------------------

def _make_input(**overrides):
    from change_order_agent.state.change_order_state import ChangeOrderInput
    defaults = dict(
        co_id="CO-001",
        project_id="PROJ-001",
        org_id="ORG-001",
        submitted_by="ABC Electrical",
        submission_timestamp=datetime.now(timezone.utc),
        contract_version="v1.0",
        cache_version="v1.0",
        raw_document="Additional conduit runs to new MRI room.",
        redacted_document="Additional conduit runs to new MRI room.",
    )
    return ChangeOrderInput(**{**defaults, **overrides})


def _make_state(**overrides):
    from change_order_agent.state.change_order_state import ChangeOrderState
    return ChangeOrderState(input=_make_input(), **overrides)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

def test_state_instantiation():
    state = _make_state()
    assert state.input.co_id == "CO-001"
    assert state.pipeline.status.value == "RUNNING"
    assert state.pipeline.awaiting_david_approval is False


def test_redaction_warning_logged(caplog):
    """State warns when redacted_document is None — raw document must never reach agents."""
    import logging
    from change_order_agent.state.change_order_state import ChangeOrderState
    with caplog.at_level(logging.WARNING):
        ChangeOrderState(input=_make_input(redacted_document=None))
    assert "redacted_document is not set" in caplog.text


def test_confidence_tier_derived_automatically():
    """Tier is always derived from score — never set manually."""
    from change_order_agent.state.change_order_state import ScopeAnalysisOutput
    assert ScopeAnalysisOutput(confidence_score=0.9).confidence_tier.value == "HIGH"
    assert ScopeAnalysisOutput(confidence_score=0.6).confidence_tier.value == "MEDIUM"
    assert ScopeAnalysisOutput(confidence_score=0.3).confidence_tier.value == "LOW"


def test_cost_range_validation_rejects_inverted_range():
    """Low cannot exceed high — Pydantic catches this immediately."""
    from change_order_agent.state.change_order_state import CostEstimationOutput
    with pytest.raises(Exception):
        CostEstimationOutput(estimated_cost_low=100_000, estimated_cost_high=50_000)


def test_valid_cost_range_accepted():
    from change_order_agent.state.change_order_state import CostEstimationOutput
    ce = CostEstimationOutput(estimated_cost_low=30_000, estimated_cost_high=50_000)
    assert ce.estimated_cost_low == 30_000


# ---------------------------------------------------------------------------
# Gating function — orchestrator logic (Step 7)
# ---------------------------------------------------------------------------

def test_gate_halts_on_low_confidence():
    from change_order_agent.graph.graph import _gate_after_scope
    from change_order_agent.state.change_order_state import ScopeAnalysisOutput, ScopeRuling
    state = _make_state(
        scope_analysis=ScopeAnalysisOutput(
            scope_ruling=ScopeRuling.AMBIGUOUS,
            confidence_score=0.3,
        )
    )
    assert _gate_after_scope(state) == "halt"


def test_gate_halts_when_pipeline_awaiting_david():
    from change_order_agent.graph.graph import _gate_after_scope
    from change_order_agent.state.change_order_state import (
        PipelineControl, PipelineStatus, ScopeAnalysisOutput, ScopeRuling,
    )
    state = _make_state(
        scope_analysis=ScopeAnalysisOutput(
            scope_ruling=ScopeRuling.IN_SCOPE,
            confidence_score=0.9,
        ),
        pipeline=PipelineControl(status=PipelineStatus.AWAITING_DAVID),
    )
    assert _gate_after_scope(state) == "halt"


def test_gate_halts_when_no_scope_ruling():
    from change_order_agent.graph.graph import _gate_after_scope
    state = _make_state()  # scope_analysis is empty by default
    assert _gate_after_scope(state) == "halt"


def test_gate_continues_on_high_confidence_in_scope():
    from change_order_agent.graph.graph import _gate_after_scope
    from change_order_agent.state.change_order_state import ScopeAnalysisOutput, ScopeRuling
    state = _make_state(
        scope_analysis=ScopeAnalysisOutput(
            scope_ruling=ScopeRuling.IN_SCOPE,
            confidence_score=0.9,
        )
    )
    assert _gate_after_scope(state) == "continue"


def test_gate_continues_on_medium_confidence():
    from change_order_agent.graph.graph import _gate_after_scope
    from change_order_agent.state.change_order_state import ScopeAnalysisOutput, ScopeRuling
    state = _make_state(
        scope_analysis=ScopeAnalysisOutput(
            scope_ruling=ScopeRuling.IN_SCOPE,
            confidence_score=0.6,
        )
    )
    assert _gate_after_scope(state) == "continue"


# ---------------------------------------------------------------------------
# Routing logic — deterministic rules table (Step 10)
# ---------------------------------------------------------------------------

def test_out_of_scope_always_routes_to_owner():
    from change_order_agent.agents.routing_agent import _determine_approver_level
    from change_order_agent.state.change_order_state import ApproverLevel, ScopeRuling
    assert _determine_approver_level(5_000, ScopeRuling.OUT_OF_SCOPE) == ApproverLevel.OWNER
    assert _determine_approver_level(500_000, ScopeRuling.OUT_OF_SCOPE) == ApproverLevel.OWNER


def test_dollar_thresholds_route_correctly():
    from change_order_agent.agents.routing_agent import _determine_approver_level
    from change_order_agent.state.change_order_state import ApproverLevel, ScopeRuling
    ruling = ScopeRuling.IN_SCOPE
    assert _determine_approver_level(10_000,  ruling) == ApproverLevel.SITE_SUPER
    assert _determine_approver_level(50_000,  ruling) == ApproverLevel.PROJECT_MANAGER
    assert _determine_approver_level(200_000, ruling) == ApproverLevel.OWNERS_REP
    assert _determine_approver_level(600_000, ruling) == ApproverLevel.OWNER


def test_unknown_work_type_defaults_to_project_manager():
    from change_order_agent.agents.routing_agent import _determine_department
    assert _determine_department(None) == "Project Manager"
    assert _determine_department("unknown work") == "Project Manager"


def test_known_work_types_map_to_correct_departments():
    from change_order_agent.agents.routing_agent import _determine_department
    assert _determine_department("electrical") == "MEP Engineer"
    assert _determine_department("mechanical") == "MEP Engineer"
    assert _determine_department("structural") == "Structural Engineer"


# ---------------------------------------------------------------------------
# Risk score — defaults to HIGH when inputs missing (Step 8)
# ---------------------------------------------------------------------------

def test_risk_score_defaults_high_when_no_scope_ruling():
    from change_order_agent.agents.output_assembly_agent import _determine_risk_score
    from change_order_agent.state.change_order_state import RiskScore
    assert _determine_risk_score(_make_state()) == RiskScore.HIGH


def test_risk_score_high_on_out_of_scope():
    from change_order_agent.agents.output_assembly_agent import _determine_risk_score
    from change_order_agent.state.change_order_state import (
        RiskScore, ScopeAnalysisOutput, ScopeRuling,
    )
    state = _make_state(
        scope_analysis=ScopeAnalysisOutput(
            scope_ruling=ScopeRuling.OUT_OF_SCOPE,
            confidence_score=0.9,
        )
    )
    assert _determine_risk_score(state) == RiskScore.HIGH


def test_risk_score_low_on_clean_in_scope():
    from change_order_agent.agents.output_assembly_agent import _determine_risk_score
    from change_order_agent.state.change_order_state import (
        CostEstimationOutput, RiskScore, ScopeAnalysisOutput, ScopeRuling,
    )
    state = _make_state(
        scope_analysis=ScopeAnalysisOutput(
            scope_ruling=ScopeRuling.IN_SCOPE,
            confidence_score=0.9,
        ),
        cost_estimation=CostEstimationOutput(
            estimated_cost_low=10_000,
            estimated_cost_high=20_000,
        ),
    )
    assert _determine_risk_score(state) == RiskScore.LOW


# ---------------------------------------------------------------------------
# Graph compilation — catches wiring errors without API calls
# ---------------------------------------------------------------------------

def test_graph_compiles_with_memory_saver():
    """Graph wires correctly — catches import errors and edge definition mistakes."""
    from langgraph.checkpoint.memory import MemorySaver
    from change_order_agent.graph.graph import build_graph
    builder = build_graph()
    compiled = builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["complete"],
    )
    assert compiled is not None

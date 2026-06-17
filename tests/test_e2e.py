"""
End-to-end test — requires a real OPENAI_API_KEY.
Run with: python -m pytest tests/test_e2e.py -v -s

What this does:
  1. Seeds the contract vector store with sample_contract.txt
  2. Seeds the historical CO vector store with sample_historical_cos.json
  3. Runs one change order through the full pipeline
  4. Prints the reviewer's report and escalation draft
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
ORG_ID = "ORG-TEST"
PROJECT_ID = "PROJ-MISSION-BAY"
CONTRACT_VERSION = "v1.0"
CACHE_VERSION = "v1.0"


# ---------------------------------------------------------------------------
# Skip if no real API key
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "").startswith("sk-smoke"),
    reason="OPENAI_API_KEY not set — skipping end-to-end test",
)


# ---------------------------------------------------------------------------
# Step 1: Seed vector stores
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def seed_stores():
    """Seed contract and historical CO stores once for the whole test session."""
    from change_order_agent.utils.seed_contract_store import seed_contract
    from change_order_agent.utils.seed_historical_store import seed_historical

    print("\n--- Seeding contract store ---")
    seed_contract(
        file_path=str(DATA_DIR / "sample_contract.txt"),
        org_id=ORG_ID,
        project_id=PROJECT_ID,
        contract_version=CONTRACT_VERSION,
    )

    print("--- Seeding historical CO store ---")
    seed_historical(
        file_path=str(DATA_DIR / "sample_historical_cos.json"),
        org_id=ORG_ID,
        project_id=PROJECT_ID,
    )
    print("--- Seeding complete ---\n")


# ---------------------------------------------------------------------------
# Step 2: Build test change order state
# ---------------------------------------------------------------------------

def _make_e2e_state(co_id: str, document: str):
    from change_order_agent.state.change_order_state import (
        ChangeOrderInput,
        ChangeOrderState,
    )
    return ChangeOrderState(
        input=ChangeOrderInput(
            co_id=co_id,
            project_id=PROJECT_ID,
            org_id=ORG_ID,
            submitted_by="Pacific Electrical Contractors Inc.",
            submission_timestamp=datetime.now(timezone.utc),
            contract_version=CONTRACT_VERSION,
            cache_version=CACHE_VERSION,
            raw_document=document,
            redacted_document=document,  # no PII in test doc
        )
    )


# ---------------------------------------------------------------------------
# Test 1: In-scope change order — should proceed through full pipeline
# ---------------------------------------------------------------------------

def test_in_scope_co_runs_full_pipeline(capsys):
    from change_order_agent.graph.run import get_current_state, process_change_order

    document = """
    CHANGE ORDER REQUEST
    Project: Mission Bay Hospital
    Subcontractor: Pacific Electrical Contractors Inc.
    Date: 2024-06-10

    Description:
    Additional electrical conduit runs and receptacle installations in the
    nurse station area on Floor 3, Wing B. Work includes 12 new duplex
    receptacles and associated conduit back to Panel 3B.

    This work was required due to revised furniture layout approved by the
    owner on May 28, 2024.

    Amount Requested: $24,500
    """

    state = _make_e2e_state("CO-E2E-001", document)

    print("\n=== RUNNING PIPELINE: CO-E2E-001 ===")
    result = process_change_order(state)

    print(f"\nPipeline status:     {result.pipeline.status.value}")
    print(f"Awaiting review:     {result.pipeline.awaiting_approval}")
    print(f"Scope ruling:        {result.scope_analysis.scope_ruling}")
    print(f"Confidence score:    {result.scope_analysis.confidence_score}")
    print(f"Confidence tier:     {result.scope_analysis.confidence_tier}")
    print(f"Cost estimate:       ${result.cost_estimation.estimated_cost_low:,.0f}–${result.cost_estimation.estimated_cost_high:,.0f}"
          if result.cost_estimation.estimated_cost_low else "Cost estimate:       not available")
    print(f"Approver level:      {result.routing.approver_level}")
    print(f"Department:          {result.routing.department}")
    print(f"Risk score:          {result.assembly.risk_score}")
    print(f"Approval stage:      {result.assembly.approval_stage}")

    if result.assembly.full_report:
        print(f"\n{'='*52}")
        print(result.assembly.full_report)

    if result.assembly.escalation_draft:
        print(f"\n{'='*52}")
        print(result.assembly.escalation_draft)

    # Assertions
    assert result.scope_analysis.scope_ruling is not None, "Scope ruling must be set"
    assert result.scope_analysis.confidence_score is not None, "Confidence score must be set"
    assert result.scope_analysis.contract_clause_cited is not None, "Contract clause must be cited"
    assert result.cost_estimation.estimated_cost_low is not None, "Cost estimate must be set"
    assert result.routing.routing_executed is True, "Routing must have executed"
    assert result.assembly.report_delivered is True, "Report must be delivered"
    assert result.assembly.escalation_draft is not None, "Escalation draft must exist"
    assert result.audit.audit_logged is True, "Audit must be logged"


# ---------------------------------------------------------------------------
# Test 2: Out-of-scope change order — should route to owner + HIGH risk
# ---------------------------------------------------------------------------

def test_out_of_scope_co_routes_to_owner(capsys):
    from change_order_agent.graph.run import process_change_order
    from change_order_agent.state.change_order_state import ApproverLevel, RiskScore

    document = """
    CHANGE ORDER REQUEST
    Project: Mission Bay Hospital
    Subcontractor: Pacific Electrical Contractors Inc.
    Date: 2024-06-11

    Description:
    Specialized power conduit installation, EMF shielding, and dedicated
    circuit panel for MRI Room 2 on Floor 2. Work includes copper shielding
    panels, specialized conduit, and 480V three-phase power feed from the
    main electrical room.

    Amount Requested: $82,000
    """

    state = _make_e2e_state("CO-E2E-002", document)

    print("\n=== RUNNING PIPELINE: CO-E2E-002 (MRI/Out-of-scope) ===")
    result = process_change_order(state)

    print(f"\nScope ruling:     {result.scope_analysis.scope_ruling}")
    print(f"Approver level:   {result.routing.approver_level}")
    print(f"Risk score:       {result.assembly.risk_score}")
    print(f"Approval stage:   {result.assembly.approval_stage}")

    assert result.assembly.risk_score == RiskScore.HIGH
    assert result.routing.approver_level == ApproverLevel.OWNER

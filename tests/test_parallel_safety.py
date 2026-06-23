"""
Parallel-safety tests — offline. They enforce the single-writer invariant that
keeps the two parallel windows collision-free: each parallel node writes ONLY
its own state section and never the shared `pipeline` control field.

If a future edit makes a parallel node write another node's field (or
`pipeline`), these tests fail loudly here — instead of the pipeline crashing at
runtime on the parallel path, where a quick test would miss it.

See the CONCURRENCY DESIGN note in state/change_order_state.py for why reducers
are deliberately not used.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _make_state(co_id: str = "CO-PAR-001"):
    from change_order_agent.state.change_order_state import (
        ChangeOrderInput, ChangeOrderState,
    )
    return ChangeOrderState(
        input=ChangeOrderInput(
            co_id=co_id,
            project_id="PROJ-001",
            org_id="ORG-001",
            submitted_by="ACME Electrical",
            submission_timestamp=datetime.now(timezone.utc),
            contract_version="v1.0",
            cache_version="v1.0",
            raw_document="raw",
            redacted_document="raw",
        )
    )


# ---------------------------------------------------------------------------
# Window 1 — retrieval + cost_estimation run concurrently after extraction
# ---------------------------------------------------------------------------

def test_window1_nodes_write_only_their_own_section(monkeypatch):
    from change_order_agent.agents import retrieval_agent, cost_estimation_agent

    # Stub the external retrieval so the nodes run offline.
    monkeypatch.setattr(retrieval_agent, "retrieve", lambda *a, **k: ["contract section text"])
    monkeypatch.setattr(cost_estimation_agent, "retrieve", lambda *a, **k: [])  # short-circuits, no LLM

    r_keys = set(retrieval_agent.run_retrieval_agent(_make_state()).keys())
    c_keys = set(cost_estimation_agent.run_cost_estimation_agent(_make_state()).keys())

    assert r_keys == {"retrieval"}
    assert c_keys == {"cost_estimation"}
    assert "pipeline" not in r_keys and "pipeline" not in c_keys
    assert r_keys.isdisjoint(c_keys)  # the two parallel siblings can never collide


# ---------------------------------------------------------------------------
# Window 2 — output_assembly + audit run concurrently after routing
# ---------------------------------------------------------------------------

def test_window2_nodes_write_only_their_own_section(monkeypatch, tmp_path):
    from change_order_agent.agents import output_assembly_agent
    import change_order_agent.utils.audit_logger as audit_logger

    # Force the template fallback (no LLM) and a temp audit DB so both run offline.
    monkeypatch.setattr(output_assembly_agent, "_try_llm_draft", lambda *a, **k: None)
    monkeypatch.setattr(audit_logger, "AUDIT_DB_PATH", str(tmp_path / "audit.db"))

    a_keys = set(output_assembly_agent.run_output_assembly_agent(_make_state()).keys())
    d_keys = set(audit_logger.run_audit_logger(_make_state()).keys())

    assert a_keys == {"assembly"}
    assert d_keys == {"audit"}
    assert "pipeline" not in a_keys and "pipeline" not in d_keys
    assert a_keys.isdisjoint(d_keys)

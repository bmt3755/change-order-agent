"""
Audit logger tests — offline (no OpenAI, no network). They write to a temp
SQLite file, never the project's audit.db. They prove the log is append-only:
re-processing a change order adds a new immutable row instead of overwriting.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _make_state(co_id: str = "CO-AUDIT-001"):
    from change_order_agent.state.change_order_state import (
        ChangeOrderInput, ChangeOrderState,
    )
    return ChangeOrderState(
        input=ChangeOrderInput(
            co_id=co_id,
            project_id="PROJ-001",
            org_id="ORG-001",
            submitted_by="Pacific Electrical Contractors Inc.",
            submission_timestamp=datetime.now(timezone.utc),
            contract_version="v1.0",
            cache_version="v1.0",
            raw_document="raw",
            redacted_document="raw",
        )
    )


def _count_rows(db_path: str, co_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE co_id = ?", (co_id,)
        ).fetchone()[0]
    finally:
        conn.close()


def test_audit_log_appends_not_overwrites(tmp_path, monkeypatch):
    """Re-processing the same CO must add a second row, never replace the first."""
    import change_order_agent.utils.audit_logger as audit
    db = str(tmp_path / "audit.db")
    monkeypatch.setattr(audit, "AUDIT_DB_PATH", db)

    state = _make_state()
    audit.run_audit_logger(state)
    audit.run_audit_logger(state)  # same co_id again

    assert _count_rows(db, state.input.co_id) == 2


def test_audit_log_id_is_unique_per_write(tmp_path, monkeypatch):
    """Each append returns its own row id — the immutable record's handle."""
    import change_order_agent.utils.audit_logger as audit
    db = str(tmp_path / "audit.db")
    monkeypatch.setattr(audit, "AUDIT_DB_PATH", db)

    state = _make_state()
    first = audit.run_audit_logger(state)["audit"]
    second = audit.run_audit_logger(state)["audit"]

    assert first.audit_logged and second.audit_logged
    assert first.audit_log_id != second.audit_log_id


def test_audit_raw_document_excluded_from_snapshot(tmp_path, monkeypatch):
    """The stored full-state JSON must not contain raw_document (PII requirement)."""
    import change_order_agent.utils.audit_logger as audit
    db = str(tmp_path / "audit.db")
    monkeypatch.setattr(audit, "AUDIT_DB_PATH", db)

    state = _make_state()
    audit.run_audit_logger(state)

    conn = sqlite3.connect(db)
    try:
        snapshot = conn.execute(
            "SELECT full_state_json FROM audit_log WHERE co_id = ?",
            (state.input.co_id,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert "raw_document" not in snapshot

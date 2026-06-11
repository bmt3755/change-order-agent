from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from langsmith import traceable

from ..state.change_order_state import AuditReference, ChangeOrderState

logger = logging.getLogger(__name__)

AUDIT_DB_PATH = os.environ.get("AUDIT_DB_PATH", "./audit.db")
SCHEMA_VERSION = "1.0"  # increment when state schema changes

# ---------------------------------------------------------------------------
# DDL — table and indexes created automatically on first run
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    co_id                 TEXT PRIMARY KEY,
    project_id            TEXT NOT NULL,
    org_id                TEXT NOT NULL,
    contract_version      TEXT NOT NULL,
    submission_timestamp  TEXT,
    logged_at             TEXT NOT NULL,
    schema_version        TEXT NOT NULL,
    scope_ruling          TEXT,
    confidence_score      REAL,
    confidence_tier       TEXT,
    contract_clause_cited TEXT,
    cost_low              REAL,
    cost_high             REAL,
    approver_level        TEXT,
    department            TEXT,
    routing_executed      INTEGER,
    risk_score            TEXT,
    approval_stage        TEXT,
    pipeline_status       TEXT,
    error_message         TEXT,
    full_state_json       TEXT NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_project        ON audit_log(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_scope_ruling   ON audit_log(scope_ruling)",
    "CREATE INDEX IF NOT EXISTS idx_risk_score     ON audit_log(risk_score)",
    "CREATE INDEX IF NOT EXISTS idx_approval_stage ON audit_log(approval_stage)",
]

_UPSERT = """
INSERT INTO audit_log (
    co_id, project_id, org_id, contract_version, submission_timestamp,
    logged_at, schema_version, scope_ruling, confidence_score, confidence_tier,
    contract_clause_cited, cost_low, cost_high, approver_level, department,
    routing_executed, risk_score, approval_stage, pipeline_status, error_message,
    full_state_json
) VALUES (
    :co_id, :project_id, :org_id, :contract_version, :submission_timestamp,
    :logged_at, :schema_version, :scope_ruling, :confidence_score, :confidence_tier,
    :contract_clause_cited, :cost_low, :cost_high, :approver_level, :department,
    :routing_executed, :risk_score, :approval_stage, :pipeline_status, :error_message,
    :full_state_json
)
ON CONFLICT(co_id) DO UPDATE SET
    logged_at             = excluded.logged_at,
    schema_version        = excluded.schema_version,
    scope_ruling          = excluded.scope_ruling,
    confidence_score      = excluded.confidence_score,
    confidence_tier       = excluded.confidence_tier,
    contract_clause_cited = excluded.contract_clause_cited,
    cost_low              = excluded.cost_low,
    cost_high             = excluded.cost_high,
    approver_level        = excluded.approver_level,
    department            = excluded.department,
    routing_executed      = excluded.routing_executed,
    risk_score            = excluded.risk_score,
    approval_stage        = excluded.approval_stage,
    pipeline_status       = excluded.pipeline_status,
    error_message         = excluded.error_message,
    full_state_json       = excluded.full_state_json
"""


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE)
    for idx in _CREATE_INDEXES:
        conn.execute(idx)


def _build_record(state: ChangeOrderState) -> dict:
    sa = state.scope_analysis
    ce = state.cost_estimation
    ro = state.routing
    ab = state.assembly
    pi = state.pipeline

    return {
        "co_id":                 state.input.co_id,
        "project_id":            state.input.project_id,
        "org_id":                state.input.org_id,
        "contract_version":      state.input.contract_version,
        "submission_timestamp":  state.input.submission_timestamp.isoformat(),
        "logged_at":             datetime.now(timezone.utc).isoformat(),
        "schema_version":        SCHEMA_VERSION,
        "scope_ruling":          sa.scope_ruling.value if sa.scope_ruling else None,
        "confidence_score":      sa.confidence_score,
        "confidence_tier":       sa.confidence_tier.value if sa.confidence_tier else None,
        "contract_clause_cited": sa.contract_clause_cited,
        "cost_low":              ce.estimated_cost_low,
        "cost_high":             ce.estimated_cost_high,
        "approver_level":        ro.approver_level.value if ro.approver_level else None,
        "department":            ro.department,
        "routing_executed":      int(ro.routing_executed),
        "risk_score":            ab.risk_score.value if ab.risk_score else None,
        "approval_stage":        ab.approval_stage.value if ab.approval_stage else None,
        "pipeline_status":       pi.status.value,
        "error_message":         pi.error_message,
        # Full snapshot — raw_document excluded; audit stores redacted version only (PII requirement)
        "full_state_json": json.dumps(
            state.model_dump(
                mode="json",
                exclude={"input": {"raw_document"}},
            )
        ),
    }


def _write_record(record: dict) -> Optional[str]:
    """Write record to SQLite inside a transaction. Returns co_id or None on failure."""
    try:
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            _init_db(conn)
            conn.execute(_UPSERT, record)
        return record["co_id"]
    except sqlite3.Error as exc:
        logger.error("Audit write failed for CO %s: %s", record.get("co_id"), exc)
        return None


# ---------------------------------------------------------------------------
# Tool node — called by LangGraph orchestrator (runs in parallel with Task 8)
# ---------------------------------------------------------------------------

@traceable(name="audit_logger")
def run_audit_logger(state: ChangeOrderState) -> dict:
    """Task 7: Log everything with full audit trail. Deterministic — no LLM."""
    co_id = state.input.co_id
    logger.info("CO %s: audit_logger starting", co_id)

    record = _build_record(state)
    audit_log_id = _write_record(record)

    # Parallel agents never write to "pipeline" — errors go to audit.error
    if audit_log_id is None:
        logger.error("CO %s: audit write failed — record not persisted", co_id)
        return {
            "audit": AuditReference(
                audit_logged=False,
                error=f"CO {co_id}: audit write failed",
            ),
        }

    logger.info("CO %s: audit record written — audit_log_id=%s", co_id, audit_log_id)

    return {
        "audit": AuditReference(
            audit_logged=True,
            audit_log_id=audit_log_id,
        ),
    }

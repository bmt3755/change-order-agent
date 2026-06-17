from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ..agents.cost_estimation_agent import run_cost_estimation_agent
from ..agents.extraction_agent import run_extraction_agent
from ..agents.output_assembly_agent import run_output_assembly_agent
from ..agents.retrieval_agent import run_retrieval_agent
from ..agents.routing_agent import run_routing_agent
from ..agents.scope_analysis_agent import run_scope_analysis_agent
from ..state.change_order_state import (
    ChangeOrderState,
    ConfidenceTier,
    PipelineStatus,
)
from ..utils.audit_logger import run_audit_logger

logger = logging.getLogger(__name__)

CHECKPOINT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "./checkpoints.db")

# ---------------------------------------------------------------------------
# Gating function — orchestrator decision after scope analysis (Step 7)
# ---------------------------------------------------------------------------

def _gate_after_scope(state: ChangeOrderState) -> str:
    """
    Implements the tiered confidence gating logic from Step 7.
    LOW confidence halts the pipeline — a human must intervene before routing.
    Any upstream AWAITING_REVIEW / FAILED / HALTED status also halts.
    """
    pi = state.pipeline
    sa = state.scope_analysis

    if pi.status in (
        PipelineStatus.AWAITING_REVIEW,
        PipelineStatus.FAILED,
        PipelineStatus.HALTED,
    ):
        return "halt"

    if sa.scope_ruling is None or sa.confidence_tier == ConfidenceTier.LOW:
        return "halt"

    return "continue"


# ---------------------------------------------------------------------------
# Completion node — runs after the reviewer approves (post-interrupt resume)
# ---------------------------------------------------------------------------

def _complete_pipeline(state: ChangeOrderState) -> dict:
    """
    Runs after Tasks 7 and 8 complete (parallel Window 2).
    Checks for errors from parallel agents, then marks the pipeline complete
    and pauses for the reviewer's approval (human-in-the-loop interrupt fires before this node).
    """
    co_id = state.input.co_id

    # Collect any errors from the parallel window
    errors = []
    if state.assembly.error:
        errors.append(state.assembly.error)
    if state.audit.error:
        errors.append(state.audit.error)

    if errors:
        logger.error("CO %s: parallel agent errors: %s", co_id, "; ".join(errors))
        return {
            "pipeline": state.pipeline.model_copy(update={
                "status": PipelineStatus.AWAITING_REVIEW,
                "current_node": "complete",
                "awaiting_approval": True,
                "error_message": "; ".join(errors),
            })
        }

    logger.info("CO %s: pipeline complete — reviewer approved", co_id)
    return {
        "pipeline": state.pipeline.model_copy(update={
            "status": PipelineStatus.COMPLETE,
            "current_node": "complete",
            "awaiting_approval": False,
            "approved_at": datetime.now(timezone.utc),
        })
    }


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(ChangeOrderState)

    # --- Nodes ---
    builder.add_node("extraction_agent",      run_extraction_agent)
    builder.add_node("retrieval_agent",        run_retrieval_agent)
    builder.add_node("cost_estimation_agent",  run_cost_estimation_agent)
    builder.add_node("scope_analysis_agent",   run_scope_analysis_agent)
    builder.add_node("routing_agent",          run_routing_agent)
    builder.add_node("output_assembly_agent",  run_output_assembly_agent)
    builder.add_node("audit_logger",           run_audit_logger)
    builder.add_node("complete",               _complete_pipeline)

    # --- Edges ---

    # Task 1 — extraction runs first
    builder.add_edge(START, "extraction_agent")

    # Tasks 2 and 4 — parallel fan-out after extraction (Step 4, Window 1)
    builder.add_edge("extraction_agent", "retrieval_agent")
    builder.add_edge("extraction_agent", "cost_estimation_agent")

    # Task 3 — scope analysis waits for both (fan-in)
    builder.add_edge("retrieval_agent",       "scope_analysis_agent")
    builder.add_edge("cost_estimation_agent", "scope_analysis_agent")

    # Conditional gate — LOW confidence or upstream failure → halt (Step 7 gatekeeper)
    builder.add_conditional_edges(
        "scope_analysis_agent",
        _gate_after_scope,
        {"continue": "routing_agent", "halt": END},
    )

    # Tasks 6, 7, 8 — parallel fan-out after routing (Step 4, Window 2)
    builder.add_edge("routing_agent", "output_assembly_agent")
    builder.add_edge("routing_agent", "audit_logger")

    # Completion node waits for both Tasks 7 and 8 (fan-in)
    builder.add_edge("output_assembly_agent", "complete")
    builder.add_edge("audit_logger",          "complete")

    builder.add_edge("complete", END)

    return builder


# ---------------------------------------------------------------------------
# Compiled app — graph + checkpointer + human-in-the-loop interrupt
# ---------------------------------------------------------------------------

def compile_app():
    """
    Compile the graph with SqliteSaver checkpointing and a pre-completion interrupt.
    The interrupt pauses before 'complete' so the reviewer can review the report and
    escalation draft before the pipeline is marked finished.
    """
    # Direct instantiation required in LangGraph 1.x — from_conn_string returns a context manager
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    builder = build_graph()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["complete"],  # human-in-the-loop pause — the reviewer reviews here
    )


app = compile_app()

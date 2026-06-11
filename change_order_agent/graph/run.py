from __future__ import annotations

import logging
from typing import Optional

from ..state.change_order_state import ChangeOrderState
from .graph import app

logger = logging.getLogger(__name__)


def _config(co_id: str) -> dict:
    """Each CO gets its own thread — checkpointer isolates state per CO."""
    return {"configurable": {"thread_id": co_id}}


# ---------------------------------------------------------------------------
# Primary entry point — runs the pipeline until the human-in-the-loop interrupt
# ---------------------------------------------------------------------------

def process_change_order(state: ChangeOrderState) -> ChangeOrderState:
    """
    Run the full pipeline for one change order.
    Stops before 'complete' node — David reviews report and draft before approving.
    Returns the paused state for David's inspection.
    """
    co_id = state.input.co_id
    logger.info("CO %s: pipeline starting", co_id)

    result = app.invoke(state, config=_config(co_id))
    final_state = ChangeOrderState(**result)

    logger.info(
        "CO %s: pipeline paused — status=%s awaiting_david=%s",
        co_id,
        final_state.pipeline.status.value,
        final_state.pipeline.awaiting_david_approval,
    )
    return final_state


# ---------------------------------------------------------------------------
# Resume after David approves
# ---------------------------------------------------------------------------

def approve_and_complete(co_id: str) -> ChangeOrderState:
    """
    Resume the pipeline after David reviews and approves the report and escalation draft.
    Runs the 'complete' node — marks the CO as fully processed.
    """
    logger.info("CO %s: David approved — resuming pipeline to completion", co_id)
    result = app.invoke(None, config=_config(co_id))
    return ChangeOrderState(**result)


# ---------------------------------------------------------------------------
# Query current state — David's dashboard view
# ---------------------------------------------------------------------------

def get_current_state(co_id: str) -> Optional[ChangeOrderState]:
    """
    Retrieve the current state of a CO pipeline without advancing it.
    Used by David's dashboard to display status, report, and draft.
    """
    snapshot = app.get_state(config=_config(co_id))
    if not snapshot.values:
        logger.warning("CO %s: no state found in checkpointer", co_id)
        return None
    return ChangeOrderState(**snapshot.values)


# ---------------------------------------------------------------------------
# Halt without completing — David rejects or overrides
# ---------------------------------------------------------------------------

def reject_and_halt(co_id: str, reason: str) -> ChangeOrderState:
    """
    Update the pipeline state to HALTED when David rejects the report or overrides the system.
    Does not resume — the CO is flagged for manual handling.
    """
    from ..state.change_order_state import PipelineStatus

    snapshot = app.get_state(config=_config(co_id))
    if not snapshot.values:
        logger.error("CO %s: cannot halt — no state found", co_id)
        raise ValueError(f"No pipeline state found for CO {co_id}")

    current = ChangeOrderState(**snapshot.values)
    updated_pipeline = current.pipeline.model_copy(update={
        "status": PipelineStatus.HALTED,
        "awaiting_david_approval": False,
        "error_message": f"Rejected by David: {reason}",
    })

    app.update_state(
        config=_config(co_id),
        values={"pipeline": updated_pipeline},
    )

    logger.info("CO %s: pipeline halted by David — reason: %s", co_id, reason)
    return get_current_state(co_id)

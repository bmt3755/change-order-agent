"""
Redaction tests — offline (no OpenAI, no network), but they DO load the
Presidio spaCy model, so they are slower than the pure smoke tests. They prove
the PII claim is real: personal data is removed, business data is kept.
"""
from __future__ import annotations

from datetime import datetime, timezone


_SAMPLE = (
    "Change order submitted by John Smith of Pacific Electrical Contractors Inc. "
    "Reach him at john.smith@pacificelec.com or call 415-555-0182. "
    "Work: additional conduit on Floor 3, Wing B near MRI Room 2, "
    "approved by the owner on May 28, 2024."
)


# ---------------------------------------------------------------------------
# redact_text — the engine
# ---------------------------------------------------------------------------

def test_personal_data_is_removed():
    """Names, emails, and phone numbers must not survive redaction."""
    from change_order_agent.utils.redaction import redact_text
    redacted, count = redact_text(_SAMPLE)

    assert "John Smith" not in redacted
    assert "john.smith@pacificelec.com" not in redacted
    assert "415-555-0182" not in redacted
    assert count >= 3  # at least person + email + phone


def test_personal_data_replaced_with_readable_tags():
    """Removed PII leaves a typed placeholder, not an empty hole."""
    from change_order_agent.utils.redaction import redact_text
    redacted, _ = redact_text(_SAMPLE)

    assert "[PERSON]" in redacted
    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted


def test_business_data_is_kept():
    """Company name, dates, and work locations are NOT personal data — keep them."""
    from change_order_agent.utils.redaction import redact_text
    redacted, _ = redact_text(_SAMPLE)

    assert "Pacific Electrical Contractors Inc." in redacted  # contract party
    assert "Floor 3" in redacted                              # work location
    assert "MRI Room 2" in redacted                           # work location
    assert "May 28, 2024" in redacted                         # business date


def test_clean_text_is_returned_unchanged():
    """Text with no PII passes through untouched, with a zero count."""
    from change_order_agent.utils.redaction import redact_text
    clean = "Additional conduit runs on Floor 3 near MRI Room 2."
    redacted, count = redact_text(clean)
    assert redacted == clean
    assert count == 0


def test_empty_text_is_safe():
    from change_order_agent.utils.redaction import redact_text
    assert redact_text("") == ("", 0)
    assert redact_text("   ") == ("   ", 0)


# ---------------------------------------------------------------------------
# run_redaction — the graph node
# ---------------------------------------------------------------------------

def _make_state(raw: str):
    from change_order_agent.state.change_order_state import (
        ChangeOrderInput, ChangeOrderState,
    )
    return ChangeOrderState(
        input=ChangeOrderInput(
            co_id="CO-RED-001",
            project_id="PROJ-001",
            org_id="ORG-001",
            submitted_by="Pacific Electrical Contractors Inc.",
            submission_timestamp=datetime.now(timezone.utc),
            contract_version="v1.0",
            cache_version="v1.0",
            raw_document=raw,
            redacted_document=None,  # the node must fill this
        )
    )


def test_node_fills_redacted_document_and_preserves_raw():
    from change_order_agent.utils.redaction import run_redaction
    state = _make_state(_SAMPLE)

    update = run_redaction(state)

    new_input = update["input"]
    assert new_input.redacted_document is not None
    assert "John Smith" not in new_input.redacted_document
    assert "[PERSON]" in new_input.redacted_document
    # raw is preserved untouched — original kept for the legal record
    assert new_input.raw_document == _SAMPLE


def test_node_fails_closed_when_redaction_raises(monkeypatch):
    """If redaction errors, the pipeline halts and redacted_document stays unset."""
    import change_order_agent.utils.redaction as redaction
    from change_order_agent.state.change_order_state import PipelineStatus

    def _boom(_text):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(redaction, "redact_text", _boom)

    state = _make_state(_SAMPLE)
    update = redaction.run_redaction(state)

    assert "input" not in update  # redacted_document never set
    assert update["pipeline"].status == PipelineStatus.FAILED

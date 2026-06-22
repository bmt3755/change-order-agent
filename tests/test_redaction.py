"""
Redaction tests — offline (no OpenAI, no network), but they DO load the
Presidio spaCy model, so they are slower than the pure smoke tests. They prove
the PII claim is real: personal data is removed, business data is kept, and the
regex backstop catches structured PII Presidio missed.
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
# redact_text — the engine (Presidio + backstop)
# ---------------------------------------------------------------------------

def test_personal_data_is_removed():
    """Names, emails, and phone numbers must not survive redaction."""
    from change_order_agent.utils.redaction import redact_text
    result = redact_text(_SAMPLE)

    assert "John Smith" not in result.text
    assert "john.smith@pacificelec.com" not in result.text
    assert "415-555-0182" not in result.text
    assert result.entities_scrubbed >= 3  # at least person + email + phone


def test_personal_data_replaced_with_readable_tags():
    """Removed PII leaves a typed placeholder, not an empty hole."""
    from change_order_agent.utils.redaction import redact_text
    text = redact_text(_SAMPLE).text

    assert "[PERSON]" in text
    assert "[EMAIL]" in text
    assert "[PHONE]" in text


def test_business_data_is_kept():
    """Company name, dates, and work locations are NOT personal data — keep them."""
    from change_order_agent.utils.redaction import redact_text
    text = redact_text(_SAMPLE).text

    assert "Pacific Electrical Contractors Inc." in text  # contract party
    assert "Floor 3" in text                              # work location
    assert "MRI Room 2" in text                           # work location
    assert "May 28, 2024" in text                         # business date


def test_clean_text_is_returned_unchanged():
    """Text with no PII passes through untouched, with zero counts."""
    from change_order_agent.utils.redaction import redact_text
    clean = "Additional conduit runs on Floor 3 near MRI Room 2."
    result = redact_text(clean)
    assert result.text == clean
    assert result.entities_scrubbed == 0
    assert result.residual_found == 0


def test_empty_text_is_safe():
    from change_order_agent.utils.redaction import redact_text
    for blank in ("", "   "):
        result = redact_text(blank)
        assert result.text == blank
        assert result.entities_scrubbed == 0
        assert result.residual_found == 0


# ---------------------------------------------------------------------------
# Backstop layer — regex sweep for structured PII Presidio missed
# ---------------------------------------------------------------------------

def test_backstop_catches_structured_pii():
    """The regex backstop scrubs email / SSN / phone shapes directly."""
    from change_order_agent.utils.redaction import _residual_scan

    email_text, n = _residual_scan("Reach me at jane.doe@example.org today.")
    assert "[EMAIL]" in email_text and "jane.doe@example.org" not in email_text and n == 1

    ssn_text, n = _residual_scan("SSN on file: 123-45-6789.")
    assert "[SSN]" in ssn_text and "123-45-6789" not in ssn_text and n == 1

    phone_text, n = _residual_scan("Call (415) 555-0147 to confirm.")
    assert "[PHONE]" in phone_text and "555-0147" not in phone_text and n == 1


def test_backstop_ignores_construction_data():
    """Dollar amounts, dates, and panel/room numbers must NOT be mistaken for PII."""
    from change_order_agent.utils.redaction import _residual_scan
    text = "Panel 3B, Revision 3, Floor 3, $24,500 approved 2024-06-10."
    scrubbed, n = _residual_scan(text)
    assert scrubbed == text
    assert n == 0


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


def test_node_fills_redacted_document_and_records_outcome():
    from change_order_agent.utils.redaction import run_redaction
    state = _make_state(_SAMPLE)

    update = run_redaction(state)

    new_input = update["input"]
    assert new_input.redacted_document is not None
    assert "John Smith" not in new_input.redacted_document
    assert "[PERSON]" in new_input.redacted_document
    # raw is preserved untouched — original kept for the legal record
    assert new_input.raw_document == _SAMPLE

    rd = update["redaction"]
    assert rd.entities_scrubbed >= 3
    assert rd.review_recommended is False  # Presidio caught it all; backstop found nothing
    assert rd.redacted_at is not None


def test_node_flags_review_when_backstop_fires(monkeypatch):
    """If the backstop catches a Presidio miss, the CO is flagged for review."""
    import change_order_agent.utils.redaction as redaction

    def _fake(_text):
        return redaction.RedactionResult(
            text="[EMAIL] survived",
            entities_scrubbed=0,
            residual_found=1,
        )

    monkeypatch.setattr(redaction, "redact_text", _fake)

    update = redaction.run_redaction(_make_state(_SAMPLE))
    rd = update["redaction"]
    assert rd.review_recommended is True
    assert rd.residual_pii_found == 1
    assert "backstop" in (rd.review_reason or "")


def test_node_fails_closed_when_redaction_raises(monkeypatch):
    """If redaction errors, the pipeline halts and redacted_document stays unset."""
    import change_order_agent.utils.redaction as redaction
    from change_order_agent.state.change_order_state import PipelineStatus

    def _boom(_text):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(redaction, "redact_text", _boom)

    update = redaction.run_redaction(_make_state(_SAMPLE))

    assert "input" not in update  # redacted_document never set
    assert update["pipeline"].status == PipelineStatus.FAILED

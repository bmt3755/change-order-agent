from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache

from langsmith import traceable
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ..state.change_order_state import ChangeOrderState, PipelineStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# What we scrub — and just as important, what we deliberately keep.
#
# REMOVE (personal data): a person's name, contact details, government and
# financial identifiers. These identify a human being and must never reach an
# agent or a third-party LLM.
#
# KEEP (business data): company / subcontractor names, dates, and physical work
# locations ("MRI Room 2", "Floor 3"). A company is a party to the contract,
# not a person; routing and reporting need its name; dates and locations are the
# actual scope of work. Redacting them would break the pipeline, not protect
# anyone — so they are intentionally absent from the list below.
#
# Each entity type maps to its own readable tag so the redacted document still
# makes sense to a human reviewer.
# ---------------------------------------------------------------------------

_ENTITY_TAGS: dict[str, str] = {
    "PERSON":            "[PERSON]",
    "EMAIL_ADDRESS":     "[EMAIL]",
    "PHONE_NUMBER":      "[PHONE]",
    "US_SSN":            "[SSN]",
    "US_ITIN":           "[TAX_ID]",
    "CREDIT_CARD":       "[CARD]",
    "US_BANK_NUMBER":    "[BANK_ACCOUNT]",
    "IBAN_CODE":         "[IBAN]",
    "US_DRIVER_LICENSE": "[DRIVER_LICENSE]",
    "US_PASSPORT":       "[PASSPORT]",
    "IP_ADDRESS":        "[IP]",
    "CRYPTO":            "[CRYPTO_WALLET]",
}

# Confidence floor. Presidio scores every match 0.0–1.0. We keep this low on
# purpose: missing a piece of PII (under-redaction) is a compliance failure,
# while flagging a borderline match (over-redaction) only makes the document
# slightly less readable. When in doubt, scrub.
SCORE_THRESHOLD = 0.30


@lru_cache(maxsize=1)
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """
    Build the Presidio engines once and reuse them.

    AnalyzerEngine loads a spaCy model (~400 MB) on first construction, which
    takes a few seconds. Caching means we pay that cost once per process, not
    once per change order.
    """
    logger.info("Loading Presidio engines (spaCy model) — first call only")
    return AnalyzerEngine(), AnonymizerEngine()


def redact_text(text: str) -> tuple[str, int]:
    """
    Replace personal data in `text` with type tags.

    Returns (redacted_text, entities_scrubbed). Runs fully offline — the text
    never leaves this machine.
    """
    if not text or not text.strip():
        return text, 0

    analyzer, anonymizer = _engines()

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=list(_ENTITY_TAGS),   # only look for what we intend to remove
        score_threshold=SCORE_THRESHOLD,
    )
    if not results:
        return text, 0

    operators = {
        entity_type: OperatorConfig("replace", {"new_value": tag})
        for entity_type, tag in _ENTITY_TAGS.items()
    }
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    return anonymized.text, len(results)


# ---------------------------------------------------------------------------
# Graph node — runs first, before any agent sees the document
# ---------------------------------------------------------------------------

@traceable(name="redaction")
def run_redaction(state: ChangeOrderState) -> dict:
    """
    Task 0: Scrub PII from raw_document into redacted_document.

    Deterministic, offline, no LLM. Fail-closed: if redaction raises, the
    pipeline is marked FAILED and redacted_document is left unset, so the
    extraction agent's guard refuses to run on raw text.
    """
    co_id = state.input.co_id
    logger.info("CO %s: redaction starting", co_id)

    try:
        redacted, scrubbed = redact_text(state.input.raw_document)
    except Exception as exc:
        # Fail-closed — never let raw text flow downstream on an error
        logger.error("CO %s: redaction failed — halting pipeline: %s", co_id, exc)
        return {
            "pipeline": state.pipeline.model_copy(update={
                "status": PipelineStatus.FAILED,
                "current_node": "redaction",
                "error_message": f"CO {co_id}: redaction failed — {exc}",
            }),
        }

    logger.info("CO %s: redaction complete — %d entities scrubbed", co_id, scrubbed)

    return {
        "input": state.input.model_copy(update={"redacted_document": redacted}),
        "pipeline": state.pipeline.model_copy(update={"current_node": "redaction"}),
    }

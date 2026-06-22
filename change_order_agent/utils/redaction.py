from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

from langsmith import traceable
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ..state.change_order_state import (
    ChangeOrderState,
    PipelineStatus,
    RedactionOutput,
)

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

# ---------------------------------------------------------------------------
# Backstop layer — runs AFTER Presidio, over the already-redacted text.
#
# These three identifiers have a fixed, unambiguous shape, so a plain regex can
# catch odd-format spans Presidio's recognizers missed. A hit here means
# Presidio let something through: we scrub it AND flag the change order for human
# review. Patterns are kept narrow on purpose so they do NOT touch construction
# data — dollar amounts ($24,500), dates (2024-06-10), or panel/room numbers.
#
# This backstop only covers STRUCTURED PII. A bare missed name with no contact
# info nearby still cannot be detected — that is an inherent limit of NER, and
# over-redaction plus human review remain the mitigation.
# ---------------------------------------------------------------------------

_RESIDUAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "US_SSN":        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE_NUMBER":  re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
    ),
}


@dataclass
class RedactionResult:
    """Outcome of redacting one document."""
    text: str
    entities_scrubbed: int = 0   # spans Presidio removed
    residual_found: int = 0      # spans the regex backstop caught after Presidio


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


def _residual_scan(text: str) -> tuple[str, int]:
    """Backstop sweep for structured PII that survived Presidio. Returns
    (scrubbed_text, hits)."""
    total = 0
    for entity_type, pattern in _RESIDUAL_PATTERNS.items():
        text, hits = pattern.subn(_ENTITY_TAGS[entity_type], text)
        total += hits
    return text, total


def redact_text(text: str) -> RedactionResult:
    """
    Replace personal data in `text` with type tags. Runs fully offline — the
    text never leaves this machine.

    Two layers: Presidio (names + contact + financial IDs), then a regex
    backstop that catches structured PII Presidio missed.
    """
    if not text or not text.strip():
        return RedactionResult(text=text)

    analyzer, anonymizer = _engines()

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=list(_ENTITY_TAGS),   # only look for what we intend to remove
        score_threshold=SCORE_THRESHOLD,
    )
    if results:
        operators = {
            entity_type: OperatorConfig("replace", {"new_value": tag})
            for entity_type, tag in _ENTITY_TAGS.items()
        }
        text = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        ).text

    text, residual = _residual_scan(text)

    return RedactionResult(
        text=text,
        entities_scrubbed=len(results),
        residual_found=residual,
    )


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

    If the regex backstop catches anything Presidio missed, the change order is
    flagged for human review (surface-only — it does not halt the pipeline).
    """
    co_id = state.input.co_id
    logger.info("CO %s: redaction starting", co_id)

    try:
        result = redact_text(state.input.raw_document)
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

    review_recommended = result.residual_found > 0
    review_reason = (
        f"{result.residual_found} PII span(s) caught by the regex backstop after "
        "Presidio — possible engine miss; verify no personal data leaked"
        if review_recommended else None
    )

    if review_recommended:
        logger.warning("CO %s: redaction review recommended — %s", co_id, review_reason)
    logger.info(
        "CO %s: redaction complete — %d scrubbed, %d backstop catches",
        co_id, result.entities_scrubbed, result.residual_found,
    )

    return {
        "input": state.input.model_copy(update={"redacted_document": result.text}),
        "redaction": RedactionOutput(
            entities_scrubbed=result.entities_scrubbed,
            residual_pii_found=result.residual_found,
            review_recommended=review_recommended,
            review_reason=review_reason,
            redacted_at=datetime.now(timezone.utc),
        ),
        "pipeline": state.pipeline.model_copy(update={"current_node": "redaction"}),
    }

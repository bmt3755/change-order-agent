# Business Impact — Architecture & Design Decisions

This documents the real architecture and design decisions in the code, and why each one
exists. Format for every decision:

- **Technical reason** — what the code does and why.
- **Business consequence** — what it changes for the people who rely on the system
  (the project manager who owns the order, the approver, the subcontractor).
- **Risk / cost impact** — the category and *direction* of impact.

> No dollar figures are claimed. The code contains no measured financial outcomes, so impact
> is stated as risk/cost categories and direction (reduces / increases), not invented numbers.

---

### 1. A single typed state object (Pydantic + enums) is the only thing passed between nodes
**Technical reason:** All inter-node data lives in one `ChangeOrderState` model with per-agent
output sections. Every controlled value (scope ruling, confidence tier, approver level, risk
score, pipeline status) is an `Enum`, and validators enforce invariants (cost low ≤ high; a
warning when the redacted document is missing). Invalid data cannot enter the state.
**Business consequence:** Downstream agents and the approver never act on malformed or
out-of-range values; bad input fails fast at the boundary instead of producing a plausible-
looking wrong report.
**Risk / cost impact:** Reduces correctness / data-integrity risk. Improves debuggability.

### 2. Confidence tier is derived from the score inside the schema, never set by hand
**Technical reason:** `ScopeAnalysisOutput` derives `HIGH / MEDIUM / LOW` from the numeric
score via a model validator (thresholds 0.75 / 0.45). There is one source of truth for tiering.
**Business consequence:** The escalation behavior tied to confidence is consistent everywhere —
no agent can disagree about what "low confidence" means.
**Risk / cost impact:** Reduces inconsistent-escalation risk. Improves reproducibility.

### 3. One conditional branch point: the tiered confidence gate
**Technical reason:** The graph has a single conditional edge, after scope analysis
(`_gate_after_scope`). It halts the pipeline on a LOW score, a missing ruling, or any upstream
`AWAITING_REVIEW / FAILED / HALTED` status; HIGH/MEDIUM continue.
**Business consequence:** An uncertain or failed scope ruling cannot silently flow into routing
and approval. Ambiguity becomes a human decision instead of an automated guess.
**Risk / cost impact:** Reduces dispute / litigation risk from wrong autonomous rulings.

### 4. Human-in-the-loop interrupt before completion, backed by per-order checkpointing
**Technical reason:** The graph compiles with `interrupt_before=["complete"]` and a SQLite
checkpointer keyed by change-order id (`thread_id`). The pipeline pauses before it is marked
complete; the human approves to resume. State survives process restarts.
**Business consequence:** Nothing is finalized without the project manager seeing the report
and the escalation draft first. A pause is not lost if the process dies mid-review.
**Risk / cost impact:** Reduces irreversible-action risk. Improves operational reliability.

### 5. The escalation email is drafted, never auto-sent
**Technical reason:** `output_assembly_agent` produces an escalation draft (LLM, with a
deterministic template fallback) and stores it for review. Every draft is tagged as requiring
the project manager's review before sending. The system has no send path.
**Business consequence:** The system is a drafting aid, not an autonomous communication
channel. A wrong or premature email never reaches an owner or subcontractor on its own.
**Risk / cost impact:** Reduces relationship / reputational risk on outbound communication.

### 6. Loose coupling: parallel agents write only their own section, never pipeline control
**Technical reason:** Agents that run inside a parallel window (retrieval, cost estimation,
assembly, audit) write only to their own output section and record failures in their own
`error` field. Only sequential nodes mutate `pipeline`. This is explicit and commented in code.
**Business consequence:** Two agents running at once cannot clobber the shared control status,
and each agent's output is attributable to that agent.
**Risk / cost impact:** Reduces concurrency / race-condition risk. Improves auditability.

### 7. Independent work runs in explicit parallel windows (fan-out / fan-in)
**Technical reason:** Extraction fans out to retrieval + cost estimation; routing fans out to
assembly + audit. Each window fans back in (scope analysis; the completion node) before the
pipeline proceeds.
**Business consequence:** Each order is processed with less wall-clock latency without changing
the result, because independent steps do not wait on each other.
**Risk / cost impact:** Improves latency / throughput per order.

### 8. Targeted RAG with multi-tenant metadata isolation and bounded retrieval
**Technical reason:** Retrieval queries are built from extracted facts and run against ChromaDB
with a `where` filter on `org_id` / `project_id` / `contract_version`, and a bounded `top_k`
(6 contract sections, 4 historical comparables).
**Business consequence:** An order can only retrieve its own organization's, project's, and
contract-version's data — and only the most relevant slice of it.
**Risk / cost impact:** Reduces confidentiality / cross-tenant-leak risk and correctness risk.
Controls token / context cost.

### 9. Deterministic, structured LLM calls with a required citation
**Technical reason:** Every LLM call uses `temperature=0` and `response_format` Pydantic
parsing. The scope analyst must return the exact `contract_clause_cited` text; a ruling without
a citation is not accepted by the schema.
**Business consequence:** Rulings are reproducible run-to-run, arrive in a validated shape, and
always carry the contract text they rest on.
**Risk / cost impact:** Reduces nondeterminism and unsupported-ruling risk. Improves auditability.

### 10. Layered failure handling: primary → fallback → flag for human
**Technical reason:** Extraction tries a full prompt (2 attempts), then a reduced prompt, then
flags for the human. Scope analysis retries then flags. Routing flags when required inputs are
missing. Assembly falls back to a template when the LLM draft fails.
**Business consequence:** A failure degrades to a simpler result or a human handoff — it never
crashes silently and never fabricates a confident answer to fill a gap.
**Risk / cost impact:** Reduces wrong-output-from-partial-failure risk. Improves reliability.

### 11. PII separation: agents see only the redacted document; the audit excludes the raw one
**Technical reason:** The input schema holds `raw_document` and `redacted_document` separately;
agents read only the redacted version, and the audit snapshot is serialized with `raw_document`
excluded. The state warns if a redacted document is missing.
**Business consequence:** Personal data is kept out of LLM calls and out of the permanent log
while still preserving a complete decision record.
**Risk / cost impact:** Reduces privacy / PII-exposure risk.

### 12. A standalone, deterministic audit logger captures the full decision record
**Technical reason:** Audit logging is its own node (not a side effect of another step). It is
LLM-free, **appends a new immutable row per run** (append-only — it never overwrites) with
indexes and a `schema_version`, and stores the full state JSON. Routing decisions are likewise
deterministic (a dollar-threshold rules table), so the record they produce is reproducible.
**Business consequence:** Every order keeps a complete, attributable, replayable **history** —
re-processing adds a new record rather than erasing the prior one — and the routing decision in
it can be re-derived from the rules.
**Risk / cost impact:** Reduces dispute-defense risk (an incomplete log loses disputes).
Improves auditability and maintainability.

### 13. Active PII redaction runs first, offline, and fails closed
**Technical reason:** A dedicated redaction node runs before any agent (the first node in the
graph). It uses Microsoft Presidio fully offline to scrub person names, emails, phones, and
government / financial identifiers from `raw_document` into `redacted_document`, replacing each
with a typed tag. Company names, dates, and work locations are deliberately kept. If redaction
raises, the node fails closed — the pipeline halts and `redacted_document` is never set, so the
extraction guard refuses to run on raw text. (This is what makes decision 11 real rather than
assumed: redaction is now performed, not just structurally separated.)
**Business consequence:** Personal data is actually removed before any agent or third-party LLM
sees the document, while the business-critical content (who the subcontractor is, where and when
the work happens) survives so the report stays usable. A redaction failure stops the order
instead of leaking the raw text.
**Risk / cost impact:** Reduces privacy / PII-exposure and compliance risk. Runs offline, so it
adds no third-party data-sharing exposure.

### 14. Defense-in-depth backstop plus a human-review flag for redaction misses
**Technical reason:** After Presidio, a narrow regex backstop sweeps the redacted text for
email / SSN / phone shapes that survived (odd-format misses), scrubs them, and counts them. The
patterns are scoped to avoid construction data (dollar amounts, dates, panel / room numbers). A
backstop hit sets `review_recommended` on a `RedactionOutput` state section and is surfaced in
the status report the reviewer already reads — surface-only, it does not halt the pipeline.
**Business consequence:** A structured-PII miss is caught and made visible to the reviewer
instead of slipping through silently, without burying the reviewer in halts that would erase the
time savings. An honest limit remains: a bare missed name with no nearby contact info cannot be
auto-detected, so over-redaction plus human review stay the mitigation.
**Risk / cost impact:** Reduces residual PII-leak risk while preserving throughput (no hard halt).

---

## Summary

| # | Decision | Where in code | Primary risk / cost impact |
|---|---|---|---|
| 1 | Single typed state (Pydantic + enums) | `state/change_order_state.py` | Correctness / data integrity ↓ |
| 2 | Confidence tier derived in-schema | `state/change_order_state.py` | Inconsistent escalation ↓ |
| 3 | Single tiered confidence gate | `graph/graph.py` `_gate_after_scope` | Dispute / litigation ↓ |
| 4 | HITL interrupt + per-order checkpointing | `graph/graph.py`, `graph/run.py` | Irreversible action ↓ · reliability ↑ |
| 5 | Escalation drafted, never auto-sent | `agents/output_assembly_agent.py` | Relationship / reputational ↓ |
| 6 | Loose coupling in parallel windows | all parallel agents | Race condition ↓ · auditability ↑ |
| 7 | Explicit parallel windows | `graph/graph.py` edges | Latency / throughput ↑ |
| 8 | Targeted RAG + tenant metadata filter | `agents/retrieval_agent.py`, `cost_estimation_agent.py` | Confidentiality ↓ · token cost ↓ |
| 9 | Deterministic structured LLM + citation | all LLM agents | Nondeterminism ↓ · auditability ↑ |
| 10 | Primary → fallback → flag handling | extraction / scope / routing / assembly | Wrong-output risk ↓ · reliability ↑ |
| 11 | PII separation (raw vs redacted) | `state/...`, `utils/audit_logger.py` | Privacy / PII exposure ↓ |
| 12 | Standalone deterministic audit + routing | `utils/audit_logger.py`, `agents/routing_agent.py` | Dispute-defense ↓ · auditability ↑ |
| 13 | Active PII redaction first, offline, fail-closed | `utils/redaction.py`, `graph/graph.py` | Privacy / PII exposure ↓ · compliance ↓ |
| 14 | Redaction backstop + human-review flag | `utils/redaction.py`, `agents/output_assembly_agent.py` | Residual PII leak ↓ · throughput preserved |

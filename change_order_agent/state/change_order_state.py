from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums — all allowed values are declared here; nothing else can be written
# ---------------------------------------------------------------------------

class ScopeRuling(str, Enum):
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS = "AMBIGUOUS"


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"      # >= 0.75 — proceed normally
    MEDIUM = "MEDIUM"  # 0.45–0.74 — proceed with flag in the reviewer's report
    LOW = "LOW"        # < 0.45 — halt pipeline, escalate to the reviewer


class ApproverLevel(str, Enum):
    SITE_SUPER = "SITE_SUPER"
    PROJECT_MANAGER = "PROJECT_MANAGER"
    OWNERS_REP = "OWNERS_REP"
    OWNER = "OWNER"


class RiskScore(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ApprovalStage(str, Enum):
    PENDING = "PENDING"
    WAITING_ON_ENGINEER = "WAITING_ON_ENGINEER"
    WAITING_ON_OWNERS_REP = "WAITING_ON_OWNERS_REP"
    WAITING_ON_OWNER = "WAITING_ON_OWNER"
    APPROVED = "APPROVED"
    ESCALATED = "ESCALATED"


class PipelineStatus(str, Enum):
    RUNNING = "RUNNING"
    AWAITING_REVIEW = "AWAITING_REVIEW"  # human-in-the-loop pause
    HALTED = "HALTED"                    # low confidence — a human must intervene
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Input — raw + redacted stored separately (PII requirement)
# ---------------------------------------------------------------------------

class ChangeOrderInput(BaseModel):
    co_id: str
    project_id: str
    org_id: str                          # for ACL isolation across GC organizations
    submitted_by: str
    submission_timestamp: datetime
    contract_version: str                # which contract version this CO is evaluated against
    cache_version: str                   # which cached reference data (contract corpus, historical COs) was used
    raw_document: str                    # original, untouched — never passed to any agent
    redacted_document: Optional[str] = None  # PII removed — the only version agents process


# ---------------------------------------------------------------------------
# Agent output sections — one per agent, all fields optional until that agent runs
# ---------------------------------------------------------------------------

class ExtractionOutput(BaseModel):
    work_type: Optional[str] = None
    subcontractor_name: Optional[str] = None
    dollar_amount_requested: Optional[float] = Field(default=None, ge=0)
    description: Optional[str] = None
    flagged_missing_amount: bool = False  # True when dollar amount was absent on the CO
    extracted_at: Optional[datetime] = None


class RetrievalOutput(BaseModel):
    contract_sections: List[str] = Field(default_factory=list)
    spec_page_references: List[str] = Field(default_factory=list)
    retrieved_at: Optional[datetime] = None
    error: Optional[str] = None  # set by parallel agent — checked by scope_analysis_agent


class ScopeAnalysisOutput(BaseModel):
    scope_ruling: Optional[ScopeRuling] = None
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_tier: Optional[ConfidenceTier] = None
    contract_clause_cited: Optional[str] = None  # exact clause text — required for legal audit trail
    reasoning: Optional[str] = None
    analyzed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def derive_confidence_tier(self) -> ScopeAnalysisOutput:
        # Tier is always derived from the score — never set manually
        if self.confidence_score is not None:
            if self.confidence_score >= 0.75:
                self.confidence_tier = ConfidenceTier.HIGH
            elif self.confidence_score >= 0.45:
                self.confidence_tier = ConfidenceTier.MEDIUM
            else:
                self.confidence_tier = ConfidenceTier.LOW
        return self


class CostEstimationOutput(BaseModel):
    estimated_cost_low: Optional[float] = Field(default=None, ge=0)
    estimated_cost_high: Optional[float] = Field(default=None, ge=0)
    historical_comparables: List[str] = Field(default_factory=list)
    estimated_at: Optional[datetime] = None
    error: Optional[str] = None  # set by parallel agent — checked by scope_analysis_agent

    @model_validator(mode="after")
    def validate_cost_range(self) -> CostEstimationOutput:
        if (
            self.estimated_cost_low is not None
            and self.estimated_cost_high is not None
            and self.estimated_cost_low > self.estimated_cost_high
        ):
            raise ValueError("estimated_cost_low cannot exceed estimated_cost_high")
        return self


class RoutingOutput(BaseModel):
    approver_name: Optional[str] = None
    approver_level: Optional[ApproverLevel] = None
    department: Optional[str] = None
    routing_executed: bool = False
    routed_at: Optional[datetime] = None


class AssemblyOutput(BaseModel):
    approval_stage: Optional[ApprovalStage] = None
    risk_score: Optional[RiskScore] = None
    full_report: Optional[str] = None      # complete report delivered to the reviewer's dashboard
    escalation_draft: Optional[str] = None  # email draft — held for review before sending
    report_delivered: bool = False
    delivered_at: Optional[datetime] = None
    error: Optional[str] = None  # set by parallel agent — checked by complete node


# ---------------------------------------------------------------------------
# Audit reference — full trail is written by the Audit Logger Tool (Chunk 7)
# ---------------------------------------------------------------------------

class AuditReference(BaseModel):
    audit_logged: bool = False
    audit_log_id: Optional[str] = None
    error: Optional[str] = None  # set by parallel agent — checked by complete node


# ---------------------------------------------------------------------------
# Pipeline control — orchestrator reads this to make gating decisions
# ---------------------------------------------------------------------------

class PipelineControl(BaseModel):
    status: PipelineStatus = PipelineStatus.RUNNING
    current_node: Optional[str] = None
    awaiting_approval: bool = False
    approved_at: Optional[datetime] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Root state — the single object LangGraph passes between every node
# ---------------------------------------------------------------------------

class ChangeOrderState(BaseModel):
    input: ChangeOrderInput
    extraction: ExtractionOutput = Field(default_factory=ExtractionOutput)
    retrieval: RetrievalOutput = Field(default_factory=RetrievalOutput)
    scope_analysis: ScopeAnalysisOutput = Field(default_factory=ScopeAnalysisOutput)
    cost_estimation: CostEstimationOutput = Field(default_factory=CostEstimationOutput)
    routing: RoutingOutput = Field(default_factory=RoutingOutput)
    assembly: AssemblyOutput = Field(default_factory=AssemblyOutput)
    audit: AuditReference = Field(default_factory=AuditReference)
    pipeline: PipelineControl = Field(default_factory=PipelineControl)

    @model_validator(mode="after")
    def warn_if_redaction_missing(self) -> ChangeOrderState:
        # Agents must never receive raw_document — warn loudly if redaction hasn't happened
        if self.input.redacted_document is None:
            logger.warning(
                "CO %s: redacted_document is not set. "
                "No agent should process raw_document directly.",
                self.input.co_id,
            )
        return self

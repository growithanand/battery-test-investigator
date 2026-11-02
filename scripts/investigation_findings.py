"""Structured generated findings and deterministic citation-reference checks."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProtocolClaim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    statement: str = Field(min_length=1, max_length=350, description="One documented protocol claim, not a measurement or causal diagnosis")
    evidence_id: str = Field(min_length=1, description="Exact ID copied from a retrieved passage")
    quote: str = Field(min_length=15, max_length=650, description="Exact contiguous supporting text copied from that passage")


class FollowUpDecision(BaseModel):
    """One evidence-dependent choice; code supplies the unchanged tool arguments."""

    model_config = ConfigDict(strict=True, extra="forbid")

    action: Literal["inspect_discharge_traces", "stop_with_partial"]
    reason: str = Field(min_length=1, max_length=350, description="Brief evidence-based reason for the action, not private reasoning")


class InvestigationFindings(BaseModel):
    # Python adds the protocol availability limitation after validating these fields.
    model_config = ConfigDict(strict=True, extra="forbid")

    protocol_claims: list[ProtocolClaim] = Field(max_length=2, description="One or two claims if evidence is available; otherwise an empty list")
    comparability: Literal["not_established"]
    physical_cause: Literal["not_determined"]
    trace_evidence_id: str | None = Field(description="Exact accepted trace evidence ID if inspection ran; otherwise null")
    limitations: list[str] = Field(min_length=1, max_length=3)
    next_check: str = Field(min_length=1, max_length=350, description="Recommended check, not a claim that it has been executed")


def finalize_findings(findings: InvestigationFindings, protocol_status: str) -> dict:
    """Preserve generated findings and add a limitation from known retrieval state."""
    protocol_limits = {
        "available": None,
        "partial": "Protocol evidence is incomplete: some applicable sources could not be retrieved or verified.",
        "unavailable": (
            "Applicable protocol evidence is unavailable. Recorded measurements alone "
            "cannot verify the intended test conditions."
        ),
    }
    if protocol_status not in protocol_limits:
        raise ValueError("Unknown protocol evidence availability status")
    return {**findings.model_dump(), "protocol_evidence_limitation": protocol_limits[protocol_status]}


RECOMMENDATION_GUIDANCE = {
    "recover_protocol_evidence": (
        "Recover and verify the unavailable applicable protocol documentation before assessing compliance."
    ),
    "review_remaining_conditions": (
        "Review active-load boundaries to distinguish active discharge from startup and post-load readings."
    ),
    "answer_stopped_request": (
        "No further check is needed for the reported-capacity comparison; "
        "broader comparability and physical cause remain unresolved."
    ),
}


def select_recommendation_guidance(protocol_status: str, action: str) -> dict:
    """Choose a recommendation focus from known evidence state and accepted action."""
    if protocol_status not in {"available", "partial", "unavailable"}:
        raise ValueError("Unknown protocol evidence availability status")
    if action == "stop_with_partial":
        focus = "answer_stopped_request"
    elif action == "inspect_discharge_traces":
        focus = "review_remaining_conditions" if protocol_status == "available" else "recover_protocol_evidence"
    else:
        raise ValueError("Unknown accepted follow-up action")
    return {"focus": focus, "next_check_hint": RECOMMENDATION_GUIDANCE[focus]}


def validate_next_check(next_check: str) -> None:
    """Reject empty text and bare workflow actions, not assess prose usefulness."""
    normalized = next_check.strip().strip(" \t\r\n\v\f`'\".").casefold()
    if not normalized or normalized in {"inspect_discharge_traces", "stop_with_partial", *RECOMMENDATION_GUIDANCE}:
        raise ValueError("Next check must explain an engineering recommendation, not a bare workflow action")


def validate_citations(findings: InvestigationFindings, protocol: dict, battery_id: str) -> None:
    """Check source identity/applicability and literal quotes, not semantic entailment."""
    passages = {item["evidence_id"]: item for item in protocol["evidence"]}
    if bool(passages) != bool(findings.protocol_claims):
        raise ValueError("Use protocol claims only when retrieved evidence is available")
    for claim in findings.protocol_claims:
        passage = passages.get(claim.evidence_id)
        if passage is None:
            raise ValueError("Citation ID was not retrieved in this investigation")
        if battery_id not in passage["applicable_batteries"]:
            raise ValueError("Cited protocol does not apply to the requested battery")
        if not claim.quote.strip() or claim.quote not in passage["text"]:
            raise ValueError("Citation quote is not an exact substring of its passage")

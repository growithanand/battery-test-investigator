"""Structured generated findings and deterministic citation-reference checks."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProtocolClaim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    statement: str = Field(min_length=1, max_length=350, description="One documented protocol claim, not a measurement or causal diagnosis")
    evidence_id: str = Field(min_length=1, description="Exact ID copied from a retrieved passage")
    quote: str = Field(min_length=15, max_length=650, description="Exact contiguous supporting text copied from that passage")


class InvestigationFindings(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    protocol_claims: list[ProtocolClaim] = Field(max_length=2, description="One or two claims if evidence is available; otherwise an empty list")
    comparability: Literal["not_established"]
    physical_cause: Literal["not_determined"]
    limitations: list[str] = Field(min_length=1, max_length=3)
    next_check: str = Field(min_length=1, max_length=350, description="Recommended check, not a claim that it has been executed")


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

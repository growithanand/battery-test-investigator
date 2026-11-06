"""Render a reviewable Markdown report from a finished investigation record."""

from html import escape
import re

from scripts.trace_inspection import format_trace_extrema


def _prose(value: str) -> str:
    """Display user and model prose without letting it change the report structure."""
    text = escape(" ".join(value.split()), quote=False)
    return re.sub(r"([\\`*_\[\]~#])", r"\\\1", text)


def render_investigation_report(result: dict) -> str:
    """Format saved evidence and the recorded outcome without executing another check."""
    run_id = result["run_id"]
    request = result["request"]
    lines = [
        "# Battery test investigation", "",
        f"- Run record: [{run_id}.json]({run_id}.json)",
        f"- Status: `{result['status']}`",
        f"- Stop reason: `{result['stop_reason']}`",
        f"- Model attempts: {result['model_attempts']}. Tool attempts: {result['tool_attempts']}.",
        "", "## Request", "", f"> {_prose(request['question'])}", "",
    ]
    battery = request["battery_id"] or "unspecified"
    early = request["early"] if request["early"] is not None else "unspecified"
    late = request["late"] if request["late"] is not None else "unspecified"
    lines.append(f"Requested battery {_prose(battery)}, earlier discharge {early}, later discharge {late}.")
    if result["clarification"]:
        lines.extend(["", f"**Clarification:** {_prose(result['clarification'])}"])

    lines.extend(["", "## Capacity comparison", ""])
    if result["comparison"] is None:
        lines.append("No capacity comparison was recorded.")
    else:
        selection = result["comparison"]["selection"]
        comparison = result["comparison"]["comparison"]
        lines.extend([
            f"Battery {selection['battery_id']}, selection revision {selection['revision']}.",
            f"MATLAB source SHA-256: `{selection['source_sha256']}`.", "",
            "| Record | Discharge | Original operation | Reported capacity (Ah) |",
            "| --- | ---: | ---: | ---: |",
            f"| Earlier | {selection['early']} | {selection['early_operation_index']} | "
            f"{comparison['earlier']['reported_capacity_Ah']:.9f} |",
            f"| Later | {selection['late']} | {selection['late_operation_index']} | "
            f"{comparison['later']['reported_capacity_Ah']:.9f} |", "",
            f"**Change (later minus earlier): {comparison['capacity_change_Ah']:+.9f} Ah "
            f"({comparison['capacity_change_percent']:+.6f}%).** The percentage uses the earlier "
            "reported capacity as its reference. Values come from the source capacity field. "
            "Test-condition comparability remains unresolved.",
        ])

    lines.extend(["", "## Protocol evidence", ""])
    protocol = result["protocol"]
    if protocol is None:
        lines.append("No protocol retrieval result was recorded.")
    else:
        lines.append(f"Retrieval status: `{protocol['status']}`.")
        if protocol["issues"]:
            lines.extend(["", "Retrieval issues:"])
            for issue in protocol["issues"]:
                document = issue.get("document_id")
                prefix = f"{_prose(document)} — " if document else ""
                lines.append(f"- {prefix}{_prose(issue['reason'])}")
        if protocol["evidence"]:
            lines.extend(["", "Retrieved evidence IDs (including document hash and line range):"])
            lines.extend(f"- {_prose(passage['evidence_id'])}" for passage in protocol["evidence"])
            lines.extend(["", "The documentation describes intended conditions. Compliance of the selected records remains unresolved."])

    lines.extend(["", "## Trace inspection", ""])
    follow_up = result["follow_up"]
    if follow_up is not None:
        lines.extend([
            f"Follow-up selected: `{follow_up['action']}`.",
            f"Question scope: `{follow_up['question_scope']}`.",
            f"Reason: {_prose(follow_up['reason'])}", "",
        ])
    evidence = result["trace_evidence"]
    if evidence is None:
        lines.append("No completed trace inspection was recorded.")
    else:
        lines.extend(["```text", format_trace_extrema(evidence), "```", "", "Trace interpretation limits:"])
        lines.extend(f"- {_prose(limit)}" for limit in evidence["limits"])

    lines.extend(["", "## Findings and recommendation", ""])
    findings = result["findings"]
    if findings is None:
        lines.append("This run has no accepted final findings or recommendation.")
    else:
        lines.extend([
            f"- Comparability: `{findings['comparability']}`.",
            f"- Physical cause: `{findings['physical_cause']}`.",
        ])
        for claim in findings["protocol_claims"]:
            lines.extend([
                "", f"**Documented claim:** {_prose(claim['statement'])}", "",
                f"Evidence ID: {_prose(claim['evidence_id'])}", "", "Source excerpt:", "",
                *(f"    {line}" for line in claim["quote"].splitlines()),
            ])
        if findings["protocol_claims"]:
            lines.extend(["", "Citation checks cover source identity, applicability and exact excerpts. "
                          "Whether a passage supports the full claim still requires review."])
        lines.extend(["", "Remaining limitations:"])
        if findings["protocol_evidence_limitation"]:
            lines.append(f"- {_prose(findings['protocol_evidence_limitation'])}")
        lines.extend(f"- {_prose(limit)}" for limit in findings["limitations"])
        lines.extend(["", f"**Recommended next check (not executed):** {_prose(findings['next_check'])}"])
    return "\n".join(lines) + "\n"

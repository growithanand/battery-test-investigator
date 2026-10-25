"""Scripted model/tool fixtures exercise the real graph, not live AI quality."""

from copy import deepcopy
import json

import pytest

pytest.importorskip("langgraph", reason="Install the optional agent dependency layer")

from langchain.tools import tool
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from scripts.investigation_findings import InvestigationFindings, validate_citations
from scripts.investigation_graph import build_investigation_graph
from scripts.model_tools import ComparisonArguments, ProtocolArguments


REQUEST = {"question": "Compare capacities and assess conditions", "battery_id": "B0005", "early": 1, "late": 2}
COMPARISON_CALL = {"name": "compare_discharge_capacity", "args": {"battery_id": "B0005", "early": 1, "late": 2}, "id": "compare-1"}
PROTOCOL_CALL = {"name": "get_battery_protocol", "args": {"battery_id": "B0005"}, "id": "protocol-1"}
QUOTE = "Discharge used a constant current of 2 A."
PROTOCOL = {
    "battery_id": "B0005", "status": "available", "issues": [],
    "evidence": [{"evidence_id": "synthetic-protocol:L1-L2", "applicable_batteries": ["B0005"],
                  "text": f"Synthetic source. {QUOTE} No physical cause established."}],
}
COMPARISON = {
    "selection": {"battery_id": "B0005", "early": 1, "late": 2, "source_sha256": "a" * 64,
                  "early_operation_index": 2, "late_operation_index": 4, "revision": 1},
    "comparison": {"capacity_change_Ah": -0.5, "capacity_change_percent": -25.0,
                   "earlier": {"reported_capacity_Ah": 2.0}, "later": {"reported_capacity_Ah": 1.5}},
}


def findings_response(protocol=PROTOCOL, **changes):
    fields = {
        "protocol_claims": [{"statement": "The documented discharge uses 2 A constant current.",
                             "evidence_id": "synthetic-protocol:L1-L2", "quote": QUOTE}] if protocol["evidence"] else [],
        "comparability": "not_established", "physical_cause": "not_determined",
        "limitations": ["Selected traces have not been inspected."],
        "next_check": "Inspect selected current and voltage traces.",
        **changes,
    }
    return {"raw": AIMessage(content=json.dumps(fields)), "parsed": InvestigationFindings(**fields), "parsing_error": None}


class StubModel:
    def __init__(self, *responses):
        self.responses, self.calls, self.bindings = list(responses), [], []

    def bind_tools(self, tools):
        self.mode = "tools"
        self.bindings.append([item.name for item in tools])
        return self

    def with_structured_output(self, schema, **kwargs):
        assert schema is InvestigationFindings and kwargs == {"method": "json_schema", "include_raw": True}
        self.mode = "findings"
        return self

    def invoke(self, messages):
        self.calls.append({"mode": self.mode, "messages": list(messages)})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def evidence_tools():
    calls = []
    results = {"comparison": deepcopy(COMPARISON), "protocol": deepcopy(PROTOCOL)}

    @tool(args_schema=ComparisonArguments)
    def compare_discharge_capacity(battery_id: str, early: int, late: int) -> dict:
        """Return deliberately constructed comparison evidence for graph tests."""
        calls.append("comparison")
        if isinstance(results["comparison"], Exception):
            raise results["comparison"]
        return deepcopy(results["comparison"])

    @tool(args_schema=ProtocolArguments)
    def get_battery_protocol(battery_id: str) -> dict:
        """Return deliberately constructed protocol evidence for graph tests."""
        calls.append("protocol")
        return deepcopy(results["protocol"])

    return (compare_discharge_capacity, get_battery_protocol), calls, results


def invoke(model, evidence_tools, request=REQUEST, **limits):
    tools, _, _ = evidence_tools
    graph = build_investigation_graph(lambda: model, *tools, **limits)
    return graph.invoke({"request": request}, {"recursion_limit": 12})


def test_both_live_shaped_tool_requests_reach_cited_partial_result(evidence_tools):
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), findings_response())
    state = invoke(model, evidence_tools)
    assert state["status"] == "partial" and state["stop_reason"] == "selected_trace_check_pending"
    assert state["model_attempts"] == 2 and state["tool_attempts"] == 2
    assert state["comparison"] == COMPARISON and state["protocol"] == PROTOCOL
    assert state["findings"]["protocol_claims"][0]["quote"] == QUOTE
    assert evidence_tools[1] == ["comparison", "protocol"]
    assert model.calls[-1]["mode"] == "findings"


@pytest.mark.parametrize("first, second", [(COMPARISON_CALL, PROTOCOL_CALL), (PROTOCOL_CALL, COMPARISON_CALL)])
def test_model_may_select_either_tool_order_within_three_attempts(evidence_tools, first, second):
    model = StubModel(AIMessage(content="", tool_calls=[first]), AIMessage(content="", tool_calls=[second]), findings_response())
    state = invoke(model, evidence_tools)
    assert state["model_attempts"] == 3 and state["findings"] is not None
    assert model.bindings[1] == [second["name"]]
    assert model.calls[1]["messages"][-1].tool_call_id == first["id"]


def test_missing_selection_stops_before_model_factory_or_tools(evidence_tools):
    tools, calls, _ = evidence_tools
    graph = build_investigation_graph(lambda: pytest.fail("Model factory must not run"), *tools)
    state = graph.invoke({"request": {"question": "Compare early and late"}})
    assert state["status"] == "needs_clarification"
    assert "battery_id, early, late" in state["clarification"]
    assert state["model_attempts"] == state["tool_attempts"] == 0 and not calls


def test_invalid_request_fails_before_model_or_tools(evidence_tools):
    state = invoke(StubModel(), evidence_tools, {**REQUEST, "early": True})
    assert state["status"] == "failed" and state["stop_reason"] == "invalid_request"
    assert not evidence_tools[1]


@pytest.mark.parametrize("decision", [
    AIMessage(content="An unsupported guessed answer"),
    AIMessage(content="", tool_calls=[{**COMPARISON_CALL, "name": "read_arbitrary_file"}]),
    AIMessage(content="", tool_calls=[{**COMPARISON_CALL, "args": {**COMPARISON_CALL["args"], "late": 3}}]),
    AIMessage(content="", tool_calls=[{**COMPARISON_CALL, "args": {**COMPARISON_CALL["args"], "early": "1"}}]),
    AIMessage(content="", tool_calls=[COMPARISON_CALL, {**COMPARISON_CALL, "id": "duplicate-tool"}]),
    AIMessage(content="", tool_calls=[COMPARISON_CALL, {**PROTOCOL_CALL, "id": COMPARISON_CALL["id"]}]),
    AIMessage(content="", tool_calls=[COMPARISON_CALL], response_metadata={"done_reason": "length"}),
])
def test_invalid_tool_batch_is_rejected_before_execution(evidence_tools, decision):
    state = invoke(StubModel(decision), evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "model_or_tool_request_error"
    assert not evidence_tools[1] and state["tool_attempts"] == 0


def test_completed_tool_cannot_be_repeated(evidence_tools):
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL]), AIMessage(content="", tool_calls=[COMPARISON_CALL]))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and evidence_tools[1] == ["comparison"]


def test_call_id_cannot_be_reused_across_batches(evidence_tools):
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL]),
                      AIMessage(content="", tool_calls=[{**PROTOCOL_CALL, "id": COMPARISON_CALL["id"]}]))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and evidence_tools[1] == ["comparison"]


def test_tool_failure_counts_attempt_and_stops_without_final_generation(evidence_tools):
    evidence_tools[2]["comparison"] = ValueError("Synthetic invalid measurements")
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "tool_error"
    assert state["tool_attempts"] == 1 and state["findings"] is None and len(model.calls) == 1


def test_tool_result_cannot_change_the_selection(evidence_tools):
    evidence_tools[2]["comparison"]["selection"]["late"] = 3
    state = invoke(StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL])), evidence_tools)
    assert state["status"] == "failed" and state["comparison"] is None


def test_inapplicable_protocol_cannot_reach_generation(evidence_tools):
    evidence_tools[2]["protocol"]["evidence"][0]["applicable_batteries"] = ["B0033"]
    model = StubModel(AIMessage(content="", tool_calls=[PROTOCOL_CALL]))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["protocol"] is None


def test_missing_protocol_produces_no_invented_document_claims(evidence_tools):
    missing = {**PROTOCOL, "status": "unavailable", "evidence": [], "issues": [{"reason": "source_missing"}]}
    evidence_tools[2]["protocol"] = missing
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), findings_response(missing))
    state = invoke(model, evidence_tools)
    assert state["status"] == "partial" and state["stop_reason"] == "protocol_evidence_incomplete"
    assert state["findings"]["protocol_claims"] == [] and state["comparison"] is not None


@pytest.mark.parametrize("limits, responses, reason", [
    ({"model_limit": 1}, [], "model_budget_exhausted"),
    ({"model_limit": 2}, [AIMessage(content="", tool_calls=[COMPARISON_CALL])], "model_budget_exhausted"),
    ({"tool_limit": 1}, [AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL])], "tool_budget_exhausted"),
])
def test_execution_budget_stops_without_extra_calls(evidence_tools, limits, responses, reason):
    model = StubModel(*responses)
    state = invoke(model, evidence_tools, **limits)
    assert state["status"] == "partial" and state["stop_reason"] == reason
    assert len(model.calls) == len(responses) and state["findings"] is None


@pytest.mark.parametrize("changes", [
    {"evidence_id": "invented-id"}, {"quote": "An invented supporting quotation."},
])
def test_bad_citation_rejected_without_retry(evidence_tools, changes):
    response = findings_response()
    response["parsed"].protocol_claims[0] = response["parsed"].protocol_claims[0].model_copy(update=changes)
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), response)
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "findings_error"
    assert state["findings"] is None and len(model.calls) == 2


def test_schema_refuses_established_comparability_or_cause():
    for fields in ({"comparability": "established"}, {"physical_cause": "SEI growth"}):
        with pytest.raises(ValidationError):
            findings_response(**fields)


def test_quote_reference_check_is_not_a_semantic_entailment_test():
    findings = findings_response()["parsed"]
    findings.protocol_claims[0].statement = "Synthetic deliberately unsupported interpretation."
    # This check only proves source/quote identity. Claim support needs separate review.
    validate_citations(findings, PROTOCOL, "B0005")


def test_model_failure_is_recorded_and_not_retried(evidence_tools):
    model = StubModel(TimeoutError("Synthetic timeout"))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["model_attempts"] == 1 and not evidence_tools[1]
    assert state["trace"][-1]["error_type"] == "TimeoutError"


def test_new_invocation_cannot_inherit_injected_old_results(evidence_tools):
    tools, calls, _ = evidence_tools
    graph = build_investigation_graph(lambda: pytest.fail("Missing request must stop"), *tools)
    state = graph.invoke({"request": {"question": "Compare"}, "comparison": COMPARISON,
                          "protocol": PROTOCOL, "findings": {"old": "result"}})
    assert state["comparison"] is state["protocol"] is state["findings"] is None and not calls


@pytest.mark.parametrize("failure", ["schema", "truncated", "extra_tool"])
def test_invalid_final_response_stops_with_no_findings_or_retry(evidence_tools, failure):
    response = findings_response()
    if failure == "schema":
        response.update(parsed=None, parsing_error=ValueError("Synthetic invalid output"))
    elif failure == "truncated":
        response["raw"] = AIMessage(content="Truncated", response_metadata={"done_reason": "length"})
    else:
        response["raw"] = AIMessage(content="", tool_calls=[COMPARISON_CALL])
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), response)
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "findings_error"
    assert state["findings"] is None and len(model.calls) == 2

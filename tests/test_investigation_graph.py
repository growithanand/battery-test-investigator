"""Scripted model/tool fixtures exercise the real graph, not live AI quality."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("langgraph", reason="Install the optional agent dependency layer")

from langchain.tools import tool
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from scripts.investigation_findings import FollowUpDecision, InvestigationFindings, validate_citations
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
TRACE_EVIDENCE = {
    "evidence_id": "synthetic-traces-1", "selection": deepcopy(COMPARISON["selection"]),
    "earlier": {"discharge_index": 1, "operation_index": 2},
    "later": {"discharge_index": 2, "operation_index": 4},
    "scope": "whole_recording", "comparability": "not_established",
}


def decision_response(action="stop_with_partial", question_scope=None):
    scope = question_scope or ("test_conditions" if action == "inspect_discharge_traces" else "reported_capacity")
    fields = {"action": action, "question_scope": scope, "reason": "Synthetic evidence-based decision for a graph test."}
    return {"raw": AIMessage(content=json.dumps(fields)), "parsed": FollowUpDecision(**fields), "parsing_error": None}


def findings_response(protocol=PROTOCOL, **changes):
    fields = {
        "protocol_claims": [{"statement": "The documented discharge uses 2 A constant current.",
                             "evidence_id": "synthetic-protocol:L1-L2", "quote": QUOTE}] if protocol["evidence"] else [],
        "comparability": "not_established", "physical_cause": "not_determined",
        "trace_evidence_id": None,
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
        assert schema in (InvestigationFindings, FollowUpDecision) and kwargs == {"method": "json_schema", "include_raw": True}
        self.mode = "findings" if schema is InvestigationFindings else "follow_up"
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
    results = {"comparison": deepcopy(COMPARISON), "protocol": deepcopy(PROTOCOL), "trace": deepcopy(TRACE_EVIDENCE)}

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

    @tool(args_schema=ComparisonArguments)
    def inspect_discharge_traces(battery_id: str, early: int, late: int) -> dict:
        """Return deliberately constructed trace evidence for graph boundary tests."""
        calls.append("trace")
        if isinstance(results["trace"], Exception):
            raise results["trace"]
        return deepcopy(results["trace"])

    return (compare_discharge_capacity, get_battery_protocol, inspect_discharge_traces), calls, results


def invoke(model, evidence_tools, request=REQUEST, **limits):
    tools, _, _ = evidence_tools
    graph = build_investigation_graph(lambda: model, *tools, **limits)
    return graph.invoke({"request": request}, {"recursion_limit": 12})


def test_both_live_shaped_tool_requests_reach_cited_partial_result(evidence_tools):
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), decision_response(), findings_response())
    state = invoke(model, evidence_tools)
    assert state["status"] == "partial" and state["stop_reason"] == "selected_trace_check_pending"
    assert state["model_attempts"] == 3 and state["tool_attempts"] == 2
    assert state["comparison"] == COMPARISON and state["protocol"] == PROTOCOL
    assert state["findings"]["protocol_claims"][0]["quote"] == QUOTE
    assert evidence_tools[1] == ["comparison", "protocol"]
    assert model.calls[-1]["mode"] == "findings"
    assert state["trace_evidence"] is None and state["follow_up"]["action"] == "stop_with_partial"


@pytest.mark.parametrize("first, second", [(COMPARISON_CALL, PROTOCOL_CALL), (PROTOCOL_CALL, COMPARISON_CALL)])
def test_model_may_select_either_tool_order_within_four_attempts(evidence_tools, first, second):
    model = StubModel(AIMessage(content="", tool_calls=[first]), AIMessage(content="", tool_calls=[second]),
                      decision_response(), findings_response())
    state = invoke(model, evidence_tools)
    assert state["model_attempts"] == 4 and state["findings"] is not None
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
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), decision_response(), findings_response(missing))
    state = invoke(model, evidence_tools)
    assert state["status"] == "partial" and state["stop_reason"] == "protocol_evidence_incomplete"
    assert state["findings"]["protocol_claims"] == [] and state["comparison"] is not None


PROTOCOL_LIMIT_CASES = [
    ("available", None),
    ("partial", "Protocol evidence is incomplete: some applicable sources could not be retrieved or verified."),
    ("unavailable", "Applicable protocol evidence is unavailable. Recorded measurements alone cannot verify the intended test conditions."),
]


@pytest.mark.parametrize("status, expected_limit", PROTOCOL_LIMIT_CASES)
@pytest.mark.parametrize("action", ["stop_with_partial", "inspect_discharge_traces"])
def test_known_protocol_limit_survives_model_omission(evidence_tools, status, expected_limit, action):
    protocol = deepcopy(PROTOCOL)
    protocol.update(status=status, issues=[] if status == "available" else [{"reason": "source_missing"}])
    if status == "unavailable":
        protocol["evidence"] = []
    evidence_tools[2]["protocol"] = protocol
    limits = ["No active-load segmentation.", "Signed current is retained.", "Statistics are not time-weighted."]
    trace_id = TRACE_EVIDENCE["evidence_id"] if action == "inspect_discharge_traces" else None
    response = findings_response(protocol, limitations=limits, trace_evidence_id=trace_id)
    original = response["parsed"].model_dump()
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response(action), response)
    state = invoke(model, evidence_tools)
    assert state["status"] == "partial"
    assert state["findings"]["protocol_evidence_limitation"] == expected_limit
    assert {key: value for key, value in state["findings"].items()
            if key != "protocol_evidence_limitation"} == original
    assert response["parsed"].model_dump() == original and state["protocol"] == protocol
    assert state["model_attempts"] == len(model.calls) == 3
    assert state["stop_reason"] == ("protocol_evidence_incomplete" if status != "available" else
                                    "comparability_assessment_pending" if trace_id else "selected_trace_check_pending")


def test_unknown_protocol_status_cannot_produce_findings(evidence_tools):
    evidence_tools[2]["protocol"]["status"] = "unexpected"
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response(), findings_response())
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "findings_error"
    assert state["findings"] is None


def test_model_cannot_supply_the_workflow_protocol_limit():
    fields = findings_response()["parsed"].model_dump()
    with pytest.raises(ValidationError):
        InvestigationFindings.model_validate({**fields, "protocol_evidence_limitation": "Invented warning."})


@pytest.mark.parametrize("status, expected_limit", PROTOCOL_LIMIT_CASES)
def test_cli_displays_protocol_and_analytical_limits(evidence_tools, monkeypatch, tmp_path, capsys, status, expected_limit):
    from scripts import run_agent_investigation as runner

    protocol = deepcopy(PROTOCOL)
    protocol["catalog_sha256"] = "c" * 64
    protocol.update(status=status, issues=[] if status == "available" else [{"reason": "source_missing"}])
    if status == "unavailable":
        protocol["evidence"] = []
    evidence_tools[2]["protocol"] = protocol
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response(), findings_response(protocol, limitations=["Constructed analytical limit."]))
    graph = build_investigation_graph(lambda: model, *evidence_tools[0])
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "build_investigation_graph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(sys, "argv", ["run_agent_investigation", "--model", "constructed-model",
                                     "--battery-id", "B0005", "--early", "1", "--late", "2"])
    runner.main()
    output = capsys.readouterr().out
    record = next((tmp_path / "outputs/agent-investigations").glob("*.json"))
    saved = json.loads(record.read_bytes())
    report = record.with_suffix(".md").read_text(encoding="utf-8")
    assert saved["findings"]["protocol_evidence_limitation"] == expected_limit
    assert "Limitation: Constructed analytical limit." in output
    assert f"Investigation report: {record.with_suffix('.md')}" in output
    assert f"]({record.name})" in report
    assert f"Retrieval status: `{status}`." in report
    assert "-0.500000000 Ah (-25.000000%)" in report
    assert "Constructed analytical limit." in report
    assert (f"    {QUOTE}" in report) == (status != "unavailable")
    if expected_limit is None:
        assert "Protocol limitation:" not in output
    else:
        assert f"Protocol limitation: {expected_limit}" in output
        assert expected_limit in report


@pytest.mark.parametrize("reject_findings", [False, True])
def test_cli_shows_executed_extrema_even_when_later_findings_fail(
        evidence_tools, monkeypatch, tmp_path, capsys, reject_findings):
    import numpy as np
    from scipy.io import savemat
    from scripts import run_agent_investigation as runner
    from scripts.model_tools import make_comparison_tool, make_trace_tool
    from scripts.trace_inspection import format_trace_extrema

    source = tmp_path / "B0005.mat"
    cycles = [{"type": "discharge", "ambient_temperature": 24,
               "data": {"Time": np.array([10, 11, 15, 21, 24]) + offset,
                        "Current_measured": np.array([0, -2, -1, 0, 0]),
                        "Voltage_measured": np.array([4.1, 3.8, 2.6, 3.4, 3.5]),
                        "Temperature_measured": np.array([24, 25, 28, 27, 26]),
                        "Capacity": capacity}}
              for offset, capacity in ((0, 2.0), (100, 1.5))]
    savemat(source, {"B0005": {"cycle": cycles}})
    trace_tool = make_trace_tool({"B0005": source})
    expected = trace_tool.invoke({"battery_id": "B0005", "early": 1, "late": 2})
    protocol = deepcopy(PROTOCOL)
    protocol["catalog_sha256"] = "c" * 64
    evidence_tools[2]["protocol"] = protocol
    response = findings_response(
        trace_evidence_id="wrong-trace-id" if reject_findings else expected["evidence_id"],
        limitations=["Whole-record inspection ran; active-load boundaries remain unresolved."],
        next_check="Review active-load boundaries before assessing comparable conditions.",
    )
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response("inspect_discharge_traces"), response)
    graph = build_investigation_graph(lambda: model, make_comparison_tool({"B0005": source}),
                                     evidence_tools[0][1], trace_tool)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "build_investigation_graph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(sys, "argv", ["run_agent_investigation", "--model", "constructed-model",
                                     "--battery-id", "B0005", "--early", "1", "--late", "2"])
    if reject_findings:
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == 2
    else:
        runner.main()
    output = capsys.readouterr().out
    record = next((tmp_path / "outputs/agent-investigations").glob("*.json"))
    saved = json.loads(record.read_bytes())
    report = record.with_suffix(".md").read_text(encoding="utf-8")
    assert saved["trace_evidence"] == expected
    assert saved["configuration"]["trace_analysis_version"] == "whole-record-traces-v2"
    assert format_trace_extrema(expected) in output
    assert output.count("| minimum |") == output.count("| maximum |") == 6
    assert f"```text\n{format_trace_extrema(expected)}\n```" in report
    assert report.count("| minimum |") == report.count("| maximum |") == 6
    assert json.loads(model.calls[-1]["messages"][-1].content)["trace_evidence"] == expected
    if reject_findings:
        assert saved["status"] == "failed" and saved["stop_reason"] == "findings_error"
        assert saved["findings"] is None and "Documented claim:" not in output
        assert "Status: `failed`" in report and "Stop reason: `findings_error`" in report
        assert "no accepted final findings or recommendation" in report
        assert "**Documented claim:**" not in report and "**Recommended next check" not in report
    else:
        assert saved["status"] == "partial" and saved["findings"]["trace_evidence_id"] == expected["evidence_id"]
        assert "Status: `partial`" in report
        assert "Review active-load boundaries before assessing comparable conditions." in report


@pytest.mark.parametrize("report_write_fails", [False, True])
def test_cli_clarification_report_preserves_saved_record(tmp_path, monkeypatch, capsys, report_write_fails):
    from scripts import run_agent_investigation as runner

    question = "Compare the discharges.\n## Fake heading <script>"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_agent_investigation", "--model", "unused-model", "--question", question])

    if report_write_fails:
        def fail_replace(self, target):
            raise OSError("Constructed report publication failure")

        monkeypatch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="Constructed report publication failure"):
            runner.main()
    else:
        runner.main()

    directory = tmp_path / "outputs/agent-investigations"
    record = next(directory.glob("*.json"))
    saved = json.loads(record.read_bytes())
    assert saved["request"]["question"] == question
    assert saved["status"] == "needs_clarification"
    assert saved["model_attempts"] == saved["tool_attempts"] == 0
    assert f"Local run record: {record}" in capsys.readouterr().out
    report = record.with_suffix(".md")
    if report_write_fails:
        assert sorted(directory.iterdir()) == [record]
    else:
        content = report.read_text(encoding="utf-8")
        assert "Status: `needs_clarification`" in content
        assert r"Please specify: battery\_id, early, late." in content
        assert r"\#\# Fake heading &lt;script&gt;" in content
        assert "**Documented claim:**" not in content and "**Recommended next check" not in content


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
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), decision_response(), response)
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "findings_error"
    assert state["findings"] is None and len(model.calls) == 3


def test_schema_refuses_established_comparability_or_cause():
    for fields in ({"comparability": "established"}, {"physical_cause": "SEI growth"}):
        with pytest.raises(ValidationError):
            findings_response(**fields)


@pytest.mark.parametrize("action", ["stop_with_partial", "inspect_discharge_traces"])
@pytest.mark.parametrize("next_check", [
    "stop_with_partial", "inspect_discharge_traces", " \n\t ", "```", " `STOP_WITH_PARTIAL`. ", "` . `",
    "recover_protocol_evidence", "review_remaining_conditions", "answer_inspected_request", "answer_stopped_request",
    "answer_extrema_request", "answer_request",
])
def test_empty_or_bare_action_recommendation_fails_without_retry(evidence_tools, action, next_check):
    inspected = action == "inspect_discharge_traces"
    response = findings_response(next_check=next_check,
                                 trace_evidence_id=TRACE_EVIDENCE["evidence_id"] if inspected else None)
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response(action), response)
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "findings_error"
    assert state["findings"] is None and state["comparison"] == COMPARISON
    assert len(model.calls) == state["model_attempts"] == 3 and not model.responses
    assert state["tool_attempts"] == (3 if inspected else 2)
    assert evidence_tools[1] == (["comparison", "protocol", "trace"] if inspected else ["comparison", "protocol"])
    assert state["trace"][-1]["error_type"] == "ValueError"


@pytest.mark.parametrize("protocol_status", ["available", "partial", "unavailable"])
def test_bounded_no_further_check_is_preserved_without_marking_investigation_complete(evidence_tools, protocol_status):
    protocol = evidence_tools[2]["protocol"]
    protocol["status"] = protocol_status
    if protocol_status != "available":
        protocol["issues"] = [{"reason": "source_missing"}]
    if protocol_status == "unavailable":
        protocol["evidence"] = []
    recommendation = (
        "No further check is needed for the reported-capacity comparison; "
        "condition comparability remains unresolved."
    )
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response(), findings_response(protocol, next_check=recommendation))
    state = invoke(model, evidence_tools, {**REQUEST, "question": "Compare only reported capacities and their change."})
    assert state["findings"]["next_check"] == recommendation
    assert state["status"] == "partial"
    assert state["stop_reason"] == ("selected_trace_check_pending" if protocol_status == "available" else "protocol_evidence_incomplete")
    assert (state["findings"]["protocol_evidence_limitation"] is None) == (protocol_status == "available")
    assert state["findings"]["comparability"] == "not_established" and state["trace_evidence"] is None
    assert state["model_attempts"] == 3 and state["tool_attempts"] == 2


@pytest.mark.parametrize("protocol_status", ["available", "partial", "unavailable"])
@pytest.mark.parametrize("action", ["stop_with_partial", "inspect_discharge_traces"])
def test_final_model_receives_code_selected_guidance_from_accepted_state(evidence_tools, protocol_status, action):
    protocol = evidence_tools[2]["protocol"]
    protocol["status"] = protocol_status
    if protocol_status != "available":
        protocol["issues"] = [{"reason": "source_missing"}]
    if protocol_status == "unavailable":
        protocol["evidence"] = []
    inspected = action == "inspect_discharge_traces"
    response = findings_response(protocol, trace_evidence_id=TRACE_EVIDENCE["evidence_id"] if inspected else None)
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), decision_response(action), response)
    state = invoke(model, evidence_tools)
    payload = json.loads(model.calls[-1]["messages"][1].content)
    expected = ("answer_stopped_request" if not inspected else
                "review_remaining_conditions" if protocol_status == "available" else "recover_protocol_evidence")
    assert payload["recommendation_guidance"]["focus"] == expected
    if inspected and protocol_status == "available":
        assert payload["recommendation_guidance"]["next_check_hint"].startswith("Review active-load boundaries")
    assert payload["recommendation_guidance"] == state["recommendation_guidance"]
    assert payload["request"] == REQUEST and payload["protocol"] == protocol
    assert payload["trace_evidence"] == (TRACE_EVIDENCE if inspected else None)
    assert state["findings"]["next_check"] == response["parsed"].next_check
    assert state["model_attempts"] == 3 and state["tool_attempts"] == (3 if inspected else 2)


@pytest.mark.parametrize("scope,action,status,focus", [
    ("recorded_extrema", "inspect_discharge_traces", "available", "answer_extrema_request"),
    ("recorded_extrema", "inspect_discharge_traces", "unavailable", "answer_extrema_request"),
    ("recorded_extrema", "stop_with_partial", "available", "answer_request"),
    ("other", "inspect_discharge_traces", "available", "answer_request"),
    ("other", "stop_with_partial", "available", "answer_request"),
    ("test_conditions", "stop_with_partial", "available", "answer_request"),
    ("reported_capacity", "inspect_discharge_traces", "available", "answer_stopped_request"),
])
def test_accepted_scope_and_executed_evidence_choose_guidance(evidence_tools, scope, action, status, focus):
    protocol = evidence_tools[2]["protocol"]
    protocol["status"] = status
    if status == "unavailable":
        protocol.update(evidence=[], issues=[{"reason": "source_missing"}])
    inspected = action == "inspect_discharge_traces"
    response = findings_response(protocol, trace_evidence_id=TRACE_EVIDENCE["evidence_id"] if inspected else None)
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response(action, scope), response)
    state = invoke(model, evidence_tools)
    assert state["follow_up"]["question_scope"] == scope
    assert state["recommendation_guidance"]["focus"] == focus
    hint = state["recommendation_guidance"]["next_check_hint"]
    if focus == "answer_request":
        assert hint is None
    elif focus == "answer_extrema_request":
        assert "No further check" in hint and "extrema and source locations" in hint
    else:
        assert "reported-capacity comparison" in hint
    payload = json.loads(model.calls[-1]["messages"][-1].content)
    assert payload["follow_up"]["question_scope"] == scope
    assert payload["recommendation_guidance"] == state["recommendation_guidance"]
    assert next(e for e in state["trace"] if e["event"] == "follow_up_selected")["question_scope"] == scope
    assert state["model_attempts"] == 3 and state["tool_attempts"] == (3 if inspected else 2)


@pytest.mark.parametrize("scope", [None, False, "segment_active_load"])
def test_follow_up_scope_rejects_missing_or_unsupported_values(scope):
    with pytest.raises(ValidationError):
        FollowUpDecision(action="inspect_discharge_traces", question_scope=scope, reason="Constructed scope test.")


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
                          "protocol": PROTOCOL, "findings": {"old": "result"},
                          "trace_evidence": TRACE_EVIDENCE, "follow_up": {"action": "inspect_discharge_traces"},
                          "recommendation_guidance": {"focus": "injected", "instruction": "Use stale guidance."}})
    assert state["comparison"] is state["protocol"] is state["findings"] is state["trace_evidence"] is state["follow_up"] is None and not calls
    assert state["recommendation_guidance"] is None


@pytest.mark.parametrize("failure", ["schema", "truncated", "extra_tool"])
def test_invalid_final_response_stops_with_no_findings_or_retry(evidence_tools, failure):
    response = findings_response()
    if failure == "schema":
        response.update(parsed=None, parsing_error=ValueError("Synthetic invalid output"))
    elif failure == "truncated":
        response["raw"] = AIMessage(content="Truncated", response_metadata={"done_reason": "length"})
    else:
        response["raw"] = AIMessage(content="", tool_calls=[COMPARISON_CALL])
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), decision_response(), response)
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "findings_error"
    assert state["findings"] is None and len(model.calls) == 3


def test_follow_up_sees_evidence_then_executes_trace_with_unchanged_arguments(evidence_tools):
    response = findings_response(trace_evidence_id=TRACE_EVIDENCE["evidence_id"],
                                 limitations=["Whole-record traces do not establish active-load boundaries."],
                                 next_check="Review active-load boundaries before assessing comparability.")
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response("inspect_discharge_traces"), response)
    state = invoke(model, evidence_tools)
    assert state["status"] == "partial" and state["stop_reason"] == "comparability_assessment_pending"
    assert state["trace_evidence"] == TRACE_EVIDENCE
    assert state["model_attempts"] == state["tool_attempts"] == 3
    assert evidence_tools[1] == ["comparison", "protocol", "trace"]
    decision_input = json.loads(model.calls[1]["messages"][-1].content)
    assert decision_input == {"request": REQUEST, "comparison": COMPARISON, "protocol": PROTOCOL}
    findings_input = json.loads(model.calls[2]["messages"][-1].content)
    assert findings_input["trace_evidence"] == TRACE_EVIDENCE
    event = next(item for item in state["trace"] if item["event"] == "follow_up_tool_requested")
    assert event["args"] == COMPARISON_CALL["args"]
    assert state["completed_tools"].count("inspect_discharge_traces") == 1


@pytest.mark.parametrize("key, value", [
    ("battery_id", "B0006"), ("early", 2), ("late", 3), ("source_sha256", "b" * 64),
    ("early_operation_index", 3), ("late_operation_index", 6), ("revision", 2),
])
def test_trace_cannot_use_changed_source_pair_mapping_or_revision(evidence_tools, key, value):
    evidence_tools[2]["trace"]["selection"][key] = value
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response("inspect_discharge_traces"))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "tool_error"
    assert state["comparison"] == COMPARISON and state["trace_evidence"] is state["findings"] is None
    assert state["tool_attempts"] == 3 and len(model.calls) == 2


@pytest.mark.parametrize("record, key", [("earlier", "operation_index"), ("later", "discharge_index")])
def test_trace_payload_must_match_its_selection_envelope(evidence_tools, record, key):
    evidence_tools[2]["trace"][record][key] = 999
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response("inspect_discharge_traces"))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["trace_evidence"] is None


def test_trace_failure_counts_attempt_and_retains_initial_evidence(evidence_tools):
    evidence_tools[2]["trace"] = OSError("Synthetic source unavailable after initial comparison")
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response("inspect_discharge_traces"))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "tool_error"
    assert state["comparison"] == COMPARISON and state["protocol"] == PROTOCOL
    assert state["tool_attempts"] == 3 and state["findings"] is None


@pytest.mark.parametrize("failure", ["schema", "truncated", "extra_tool", "timeout", "unknown_action", "changed_selection"])
def test_invalid_follow_up_stops_before_trace_or_findings(evidence_tools, failure):
    response = decision_response("inspect_discharge_traces")
    if failure == "schema":
        response.update(parsed=None, parsing_error=ValueError("Synthetic invalid output"))
    elif failure == "truncated":
        response["raw"] = AIMessage(content="Truncated", response_metadata={"done_reason": "length"})
    elif failure == "extra_tool":
        response["raw"] = AIMessage(content="", tool_calls=[COMPARISON_CALL])
    elif failure == "timeout":
        response = TimeoutError("Synthetic follow-up timeout")
    else:
        response["parsed"] = {"action": "read_arbitrary_file" if failure == "unknown_action" else "inspect_discharge_traces",
                              "reason": "Synthetic invalid action."}
        if failure == "changed_selection":
            response["parsed"]["late"] = 3
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), response)
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "follow_up_error"
    assert evidence_tools[1] == ["comparison", "protocol"]
    assert state["model_attempts"] == 2 and state["trace_evidence"] is state["findings"] is None


@pytest.mark.parametrize("model_limit, tool_limit, responses, reason", [
    (2, 3, [], "model_budget_exhausted"),
    (4, 2, [decision_response("inspect_discharge_traces")], "tool_budget_exhausted"),
])
def test_follow_up_preserves_findings_reservation_and_tool_limit(evidence_tools, model_limit, tool_limit, responses, reason):
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]), *responses)
    state = invoke(model, evidence_tools, model_limit=model_limit, tool_limit=tool_limit)
    assert state["status"] == "partial" and state["stop_reason"] == reason
    assert state["tool_attempts"] == 2 and evidence_tools[1] == ["comparison", "protocol"]
    assert state["findings"] is None and len(model.calls) == 1 + len(responses)


@pytest.mark.parametrize("action, evidence_id", [
    ("stop_with_partial", TRACE_EVIDENCE["evidence_id"]),
    ("inspect_discharge_traces", None), ("inspect_discharge_traces", "invented-traces"),
])
def test_findings_cannot_invent_omit_or_change_trace_reference(evidence_tools, action, evidence_id):
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response(action), findings_response(trace_evidence_id=evidence_id))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["stop_reason"] == "findings_error"
    assert state["findings"] is None


def test_missing_protocol_does_not_prevent_useful_trace_inspection(evidence_tools):
    missing = {**PROTOCOL, "status": "unavailable", "evidence": [], "issues": [{"reason": "source_missing"}]}
    evidence_tools[2]["protocol"] = missing
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response("inspect_discharge_traces"),
                      findings_response(missing, trace_evidence_id=TRACE_EVIDENCE["evidence_id"]))
    state = invoke(model, evidence_tools)
    assert state["status"] == "partial" and state["stop_reason"] == "protocol_evidence_incomplete"
    assert state["trace_evidence"] == TRACE_EVIDENCE and state["findings"]["protocol_claims"] == []


def test_separate_initial_calls_still_allow_one_trace_check_and_final_findings(evidence_tools):
    model = StubModel(AIMessage(content="", tool_calls=[PROTOCOL_CALL]),
                      AIMessage(content="", tool_calls=[COMPARISON_CALL]),
                      decision_response("inspect_discharge_traces"),
                      findings_response(trace_evidence_id=TRACE_EVIDENCE["evidence_id"]))
    state = invoke(model, evidence_tools)
    assert state["status"] == "partial" and state["findings"] is not None
    assert state["model_attempts"] == 4 and state["tool_attempts"] == 3


def test_stop_decision_can_produce_findings_with_no_remaining_tool_budget(evidence_tools):
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response(), findings_response())
    state = invoke(model, evidence_tools, tool_limit=2)
    assert state["findings"] is not None and state["trace_evidence"] is None
    assert state["tool_attempts"] == 2 and evidence_tools[1] == ["comparison", "protocol"]


@pytest.mark.parametrize("evidence_id", ["", "   "])
def test_trace_without_evidence_identity_is_rejected(evidence_tools, evidence_id):
    evidence_tools[2]["trace"]["evidence_id"] = evidence_id
    model = StubModel(AIMessage(content="", tool_calls=[COMPARISON_CALL, PROTOCOL_CALL]),
                      decision_response("inspect_discharge_traces"))
    state = invoke(model, evidence_tools)
    assert state["status"] == "failed" and state["trace_evidence"] is None

"""Constructed records check false-pass boundaries, not live model quality."""

from copy import deepcopy
import hashlib
import json
import sys

import pytest

from scripts import evaluate_follow_up as evaluation


CASE_SET = json.loads(evaluation.CASES_PATH.read_bytes())
CASES = {case["case_id"]: case for case in CASE_SET["cases"]}


def make_record(case_id="trace-1-168"):
    case = CASES[case_id]
    request, expected = case["request"], case["expected"]
    record = {
        "run_id": "constructed-evaluation-record", "request": deepcopy(request),
        "status": expected["status"], "stop_reason": expected["stop_reason"],
        "clarification": "Specify battery_id, early and late.",
        "follow_up": None, "trace_evidence": None, "comparison": None,
        "protocol": None, "findings": None, "completed_tools": [], "pending_calls": [],
        "model_attempts": 0, "tool_attempts": 0, "trace": [],
        "configuration": {"limits": deepcopy(CASE_SET["limits"]), "model_used": None,
                          "workflow_version": "constructed-fixture"},
        "input_configuration_sha256": "constructed-config-identity",
    }
    if expected["action"] is None:
        return record
    source = CASE_SET["source_identity"]
    selection = {
        "battery_id": request["battery_id"], "source_sha256": source["source_sha256"],
        "early": request["early"], "late": request["late"], "revision": 1,
        "early_operation_index": source["operation_indices"][str(request["early"])],
        "late_operation_index": source["operation_indices"][str(request["late"])],
    }
    record.update({
        "comparison": {"selection": selection},
        "protocol": {"battery_id": request["battery_id"], "status": expected["protocol_status"],
                     "evidence": [], "issues": [{"reason": "constructed-source-missing"}]},
        "follow_up": {"action": expected["action"], "reason": "Constructed evidence-gap explanation."},
        "findings": {"comparability": "not_established", "physical_cause": "not_determined",
                     "trace_evidence_id": None, "protocol_claims": [], "limitations": ["Constructed limit."]},
        "model_attempts": 3,
    })
    calls = [
        {"name": "compare_discharge_capacity", "args": {key: request[key] for key in ("battery_id", "early", "late")}, "id": "compare-1"},
        {"name": "get_battery_protocol", "args": {"battery_id": request["battery_id"]}, "id": "protocol-1"},
    ]
    record["trace"] = [
        {"event": "tools_requested", "calls": calls},
        *({"event": "tool_returned", "name": call["name"], "call_id": call["id"]} for call in calls),
        {"event": "follow_up_selected", **record["follow_up"]},
    ]
    record["completed_tools"] = [call["name"] for call in calls]
    if expected["trace_required"]:
        record["trace_evidence"] = {
            "evidence_id": "constructed-trace-id", "selection": deepcopy(selection),
            "earlier": {"discharge_index": request["early"], "operation_index": selection["early_operation_index"]},
            "later": {"discharge_index": request["late"], "operation_index": selection["late_operation_index"]},
        }
        record["findings"]["trace_evidence_id"] = "constructed-trace-id"
        record["completed_tools"].append(evaluation.TRACE_TOOL)
        record["trace"].extend([
            {"event": "follow_up_tool_requested", "name": evaluation.TRACE_TOOL, "args": deepcopy(calls[0]["args"])},
            {"event": "tool_returned", "name": evaluation.TRACE_TOOL, "evidence_id": "constructed-trace-id"},
        ])
    record["tool_attempts"] = len(record["completed_tools"])
    return record


def assess(record, case_id="trace-1-168"):
    return evaluation.evaluate_follow_up(record, CASES[case_id], CASE_SET)


@pytest.mark.parametrize("case_id", [case["case_id"] for case in CASE_SET["cases"] if case["split"] == "development"])
def test_valid_execution_contract_still_requires_human_review(case_id):
    report = assess(make_record(case_id), case_id)
    assert report["automated_status"] == "passed" and report["failures"] == []
    assert report["case_outcome"] == "human_review_required" and report["review_status"] == "pending"
    assert report["assessment_type"] == "saved_record_review"


@pytest.mark.parametrize("path, value", [
    (("request", "question"), "Another question"),
    (("request", "early"), True),
    (("status",), "complete"),
    (("stop_reason",), "selected_trace_check_pending"),
    (("follow_up", "action"), "stop_with_partial"),
    (("comparison", "selection", "source_sha256"), "b" * 64),
    (("trace_evidence", "selection", "revision"), 2),
    (("trace_evidence", "later", "operation_index"), 352),
    (("trace_evidence", "earlier", "discharge_index"), True),
    (("trace_evidence",), None),
    (("trace_evidence", "evidence_id"), ""),
    (("findings", "trace_evidence_id"), "another-trace-id"),
    (("findings", "physical_cause"), "Invented diagnosis"),
    (("findings", "comparability"), "established"),
    (("configuration", "limits", "tool_limit"), 999),
    (("tool_attempts",), 2),
    (("model_attempts",), True),
    (("pending_calls",), [{"name": "unfinished"}]),
    (("trace",), []),
    (("completed_tools",), ["compare_discharge_capacity"]),
])
def test_changed_request_evidence_or_execution_cannot_pass(path, value):
    record = make_record()
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert assess(record)["automated_status"] == "failed"


@pytest.mark.parametrize("change", ["changed_pair", "duplicate_call_id", "wrong_return_id", "wrong_trace_id", "changed_decision"])
def test_trace_of_actual_calls_must_agree_with_summary_fields(change):
    record = make_record()
    if change == "changed_pair":
        record["trace"][-2]["args"]["late"] = 100
    elif change == "duplicate_call_id":
        record["trace"][0]["calls"][1]["id"] = "compare-1"
    elif change == "wrong_return_id":
        record["trace"][1]["call_id"] = "not-the-requested-id"
    elif change == "wrong_trace_id":
        record["trace"][-1]["evidence_id"] = "not-the-accepted-trace"
    else:
        record["trace"][3]["action"] = "stop_with_partial"
    assert "tool_trace_matches_execution" in assess(record)["failures"]


def test_missing_protocol_cannot_have_substitute_claims():
    record = make_record("missing-protocol-1-100")
    record["findings"]["protocol_claims"] = [{"statement": "Invented protocol."}]
    assert "missing_protocol_not_replaced_with_claims" in assess(record, "missing-protocol-1-100")["failures"]


@pytest.mark.parametrize("change", ["model_attempt", "evidence", "model_loaded", "recorded_call"])
def test_clarification_cannot_hide_execution(change):
    record = make_record("missing-selection")
    if change == "model_attempt":
        record["model_attempts"] = 1
    elif change == "evidence":
        record["comparison"] = {"old": "evidence"}
    elif change == "model_loaded":
        record["configuration"]["model_used"] = {"model": "unexpected"}
    else:
        record["trace"] = [{"event": "tool_returned", "name": "compare_discharge_capacity", "call_id": "old"}]
    assert assess(record, "missing-selection")["automated_status"] == "failed"


@pytest.mark.parametrize("scope", [False, "unsupported_scope", []])
def test_unknown_declared_scope_cannot_pass_when_trace_and_record_agree(scope):
    record = make_record()
    record["follow_up"]["question_scope"] = scope
    next(e for e in record["trace"] if e["event"] == "follow_up_selected")["question_scope"] = scope
    assert "declared_question_scope_is_known" in assess(record)["failures"]


def test_saved_scope_must_match_the_selected_decision_event():
    record = make_record()
    record["follow_up"]["question_scope"] = "recorded_extrema"
    event = next(e for e in record["trace"] if e["event"] == "follow_up_selected")
    event["question_scope"] = "recorded_extrema"
    assert assess(record)["automated_status"] == "passed"
    event["question_scope"] = "test_conditions"
    assert "tool_trace_matches_execution" in assess(record)["failures"]


def test_numeric_accuracy_and_free_text_meaning_are_not_claimed_by_this_checker():
    record = make_record()
    record["comparison"]["comparison"] = {"capacity_change_Ah": 999}
    record["findings"]["limitations"] = ["Deliberately unsupported semantic claim."]
    report = assess(record)
    assert report["automated_status"] == "passed" and report["case_outcome"] == "human_review_required"
    assert "Capacity arithmetic and trace measurement accuracy" in report["not_evaluated"]


def test_cli_reports_file_identities_and_does_not_modify_inputs(tmp_path, monkeypatch, capsys):
    path = tmp_path / "constructed-run.json"
    data = json.dumps(make_record()).encode()
    path.write_bytes(data)
    monkeypatch.setattr(sys, "argv", ["evaluate_follow_up", "--case", "trace-1-168", "--record", str(path)])
    evaluation.main()
    report = json.loads(capsys.readouterr().out)
    assert report["record_sha256"] == hashlib.sha256(data).hexdigest()
    assert report["case_set_sha256"] == hashlib.sha256(evaluation.CASES_PATH.read_bytes()).hexdigest()
    assert path.read_bytes() == data and report["case_outcome"] == "human_review_required"


@pytest.mark.parametrize("failure, code", [("wrong_decision", 1), ("unknown_case", 2), ("invalid_json", 2), ("missing_fields", 2)])
def test_cli_does_not_report_pass_for_failures_or_invalid_inputs(tmp_path, monkeypatch, capsys, failure, code):
    path = tmp_path / "constructed-run.json"
    record = make_record()
    if failure == "wrong_decision":
        record["follow_up"]["action"] = "stop_with_partial"
    path.write_text("not JSON" if failure == "invalid_json" else json.dumps({} if failure == "missing_fields" else record))
    case_id = "unknown" if failure == "unknown_case" else "trace-1-168"
    monkeypatch.setattr(sys, "argv", ["evaluate_follow_up", "--case", case_id, "--record", str(path)])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == code
    report = json.loads(capsys.readouterr().out)
    assert report.get("automated_status") != "passed"

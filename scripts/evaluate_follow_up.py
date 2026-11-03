"""Check one saved investigation against a fixed follow-up case, without inference."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evaluations/follow-up-cases.json"
INITIAL_TOOLS = {"compare_discharge_capacity", "get_battery_protocol"}
TRACE_TOOL = "inspect_discharge_traces"


def same_json(left, right):
    """Keep JSON types distinct: True and 1 must not identify the same selection."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def evaluate_follow_up(record: dict, case: dict, case_set: dict) -> dict:
    """Check action/execution contracts; do not claim numerical or semantic correctness."""
    required = {"request", "follow_up", "trace_evidence", "comparison", "protocol", "findings",
                "status", "stop_reason", "model_attempts", "tool_attempts", "completed_tools",
                "pending_calls", "trace", "configuration", "run_id", "input_configuration_sha256"}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("Expected a saved agent investigation record with evidence and execution metadata")
    expected, request = case["expected"], case["request"]
    decision, evidence = record["follow_up"], record["trace_evidence"]
    checks = {
        "request_matches": same_json(record["request"], request),
        "stopping_matches": (record["status"], record["stop_reason"]) == (expected["status"], expected["stop_reason"]),
        "decision_matches": (decision.get("action") if isinstance(decision, dict) else None) == expected["action"],
        "trace_presence_matches": isinstance(evidence, dict) if expected["trace_required"] else evidence is None,
        "no_pending_calls_or_raw_messages": record["pending_calls"] == [] and "messages" not in record,
        "tool_trace_matches_execution": tool_trace_matches(record, request),
    }
    if isinstance(decision, dict) and "question_scope" in decision:
        checks["declared_question_scope_is_known"] = type(decision["question_scope"]) is str and decision["question_scope"] in {
            "reported_capacity", "recorded_extrema", "test_conditions", "other",
        }
    limits = case_set["limits"]
    checks["attempts_within_limits"] = all(
        type(record[field]) is int and 0 <= record[field] <= limits[limit]
        for field, limit in (("model_attempts", "model_limit"), ("tool_attempts", "tool_limit"))
    )
    checks["declared_limits_match"] = same_json(record["configuration"]["limits"], limits)
    if expected["action"] is None:
        checks["stopped_before_evidence_or_model"] = (
            all(record[key] is None for key in ("comparison", "protocol", "findings", "follow_up", "trace_evidence"))
            and record["completed_tools"] == [] and record["model_attempts"] == record["tool_attempts"] == 0
            and record["configuration"].get("model_used") is None
        )
        if expected["status"] == "needs_clarification":
            checks["clarification_present"] = bool(str(record.get("clarification") or "").strip())
    else:
        selection = record["comparison"]["selection"] if isinstance(record["comparison"], dict) else None
        source = case_set["source_identity"]
        expected_selection = {
            "battery_id": source["battery_id"], "source_sha256": source["source_sha256"],
            "early": request["early"], "late": request["late"], "revision": source["revision"],
            "early_operation_index": source["operation_indices"][str(request["early"])],
            "late_operation_index": source["operation_indices"][str(request["late"])],
        }
        checks["comparison_selection_matches_reference"] = same_json(selection, expected_selection)
        completed = record["completed_tools"]
        needed = INITIAL_TOOLS | ({TRACE_TOOL} if expected["trace_required"] else set())
        checks["completed_checks_match"] = (
            isinstance(completed, list) and len(completed) == len(needed) and set(completed) == needed
            and record["tool_attempts"] == len(needed) and 3 <= record["model_attempts"] <= limits["model_limit"]
        )
        protocol, findings = record["protocol"], record["findings"]
        checks["protocol_availability_matches"] = (
            isinstance(protocol, dict) and protocol.get("status") == expected["protocol_status"]
            and protocol.get("battery_id") == request["battery_id"]
        )
        checks["unresolved_limits_retained"] = (
            isinstance(findings, dict) and findings.get("comparability") == "not_established"
            and findings.get("physical_cause") == "not_determined"
        )
        checks["missing_protocol_not_replaced_with_claims"] = (
            expected["protocol_status"] != "unavailable" or (
                isinstance(protocol, dict) and protocol.get("evidence") == [] and bool(protocol.get("issues"))
                and isinstance(findings, dict) and findings.get("protocol_claims") == []
            )
        )
        if expected["trace_required"]:
            checks["trace_identity_matches"] = (
                isinstance(evidence, dict) and same_json(evidence.get("selection"), expected_selection)
                and all(same_json(evidence.get(key, {}).get("discharge_index"), expected_selection[index])
                        and same_json(evidence.get(key, {}).get("operation_index"), expected_selection[operation])
                        for key, index, operation in (("earlier", "early", "early_operation_index"),
                                                      ("later", "late", "late_operation_index")))
            )
        checks["findings_reference_executed_trace"] = (
            isinstance(findings, dict) and findings.get("trace_evidence_id") == (evidence.get("evidence_id") if isinstance(evidence, dict) else None)
            and (evidence is None or isinstance(evidence.get("evidence_id"), str) and bool(evidence["evidence_id"].strip()))
        )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "assessment_type": "saved_record_review", "case_set_version": case_set["case_set_version"],
        "case_id": case["case_id"], "split": case["split"], "run_id": record["run_id"],
        "automated_status": "failed" if failures else "passed", "checks": checks, "failures": failures,
        "case_outcome": "automated_checks_failed" if failures else "human_review_required",
        "review_status": "pending", "manual_review": case_set["manual_review"],
        "review_material": {"decision": decision, "findings": record["findings"], "protocol": record["protocol"]},
        "configuration": record["configuration"], "input_configuration_sha256": record["input_configuration_sha256"],
        "repeat_plan": case_set["repeat_plan"],
        "not_evaluated": ["Capacity arithmetic and trace measurement accuracy", "Original source bytes and citation meaning",
                          "Decision reason or other free-text semantic support", "Repeated-run reliability or release completion"],
    }


def tool_trace_matches(record, request):
    """Verify recorded calls/returns as well as the final completed-tools list."""
    requested, returned, ids, decisions = [], [], [], []
    call_ids = {}
    for event in record["trace"]:
        if event["event"] == "tools_requested":
            for call in event["calls"]:
                if call["name"] not in INITIAL_TOOLS:
                    return False
                expected = {"battery_id": request["battery_id"]}
                if call["name"] == "compare_discharge_capacity":
                    expected.update(early=request["early"], late=request["late"])
                if not same_json(call["args"], expected):
                    return False
                requested.append(call["name"])
                ids.append(call["id"])
                call_ids[call["name"]] = call["id"]
        elif event["event"] == "follow_up_selected":
            keys = ("action", "reason", "question_scope") if "question_scope" in event else ("action", "reason")
            decisions.append({key: event[key] for key in keys})
        elif event["event"] == "follow_up_tool_requested":
            if event["name"] != TRACE_TOOL or not same_json(event["args"], {key: request[key] for key in ("battery_id", "early", "late")}):
                return False
            requested.append(TRACE_TOOL)
        elif event["event"] == "tool_returned":
            if event["name"] in INITIAL_TOOLS and event.get("call_id") != call_ids.get(event["name"]):
                return False
            if event["name"] == TRACE_TOOL and event.get("evidence_id") != (record["trace_evidence"] or {}).get("evidence_id"):
                return False
            returned.append(event["name"])
    return (requested == returned == record["completed_tools"] and len(ids) == len(set(ids))
            and all(isinstance(item, str) and bool(item) for item in ids)
            and same_json(decisions, [record["follow_up"]] if record["follow_up"] is not None else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path, help="Existing local agent run JSON; never invokes a model")
    parser.add_argument("--case", required=True, help="Exact case ID from the versioned case set")
    args = parser.parse_args()
    try:
        case_bytes = CASES_PATH.read_bytes()
        case_set = json.loads(case_bytes)
        case = next((item for item in case_set["cases"] if item["case_id"] == args.case), None)
        if case is None:
            raise ValueError(f"Unknown follow-up case: {args.case}")
        record_bytes = args.record.read_bytes()
        report = evaluate_follow_up(json.loads(record_bytes), case, case_set)
        report["case_set_sha256"] = hashlib.sha256(case_bytes).hexdigest()
        report["record_sha256"] = hashlib.sha256(record_bytes).hexdigest()
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({"assessment_type": "saved_record_review", "error": str(error)}))
        raise SystemExit(2) from error
    if report["automated_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

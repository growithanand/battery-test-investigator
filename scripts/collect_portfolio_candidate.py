"""Run the current local investigator on the 24 frozen development cases."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

from scripts.collect_portfolio_baseline import CASES, ROOT, git, sha, write_json
from scripts.evaluate_follow_up import evaluate_follow_up, same_json


BASELINE = ROOT / "evaluations/baselines/portfolio-d7e314f-20261107.json"
SELECTION_FIELDS = ("battery_id", "early", "late")
REJECTION_REASONS = {
    "invalid_request": "invalid_request",
    "out_of_range": "tool_error",
    "unavailable_model": "model_or_tool_request_error",
}


def read_inputs():
    """Use the published baseline reference and its original fixture bytes."""
    baseline_bytes, case_bytes = BASELINE.read_bytes(), CASES.read_bytes()
    baseline, suite = json.loads(baseline_bytes), json.loads(case_bytes)
    baseline_plan = ROOT / baseline["evidence"]["local_directory"] / "plan.json"
    rubric = ROOT / baseline["rubric"]
    if not baseline_plan.is_file():
        raise ValueError("The frozen local baseline plan and its protocol fixtures are required")
    plan_bytes, rubric_bytes = baseline_plan.read_bytes(), rubric.read_bytes()
    if (sha(case_bytes) != baseline["case_set_sha256"] or sha(rubric_bytes) != baseline["rubric_sha256"]
            or sha(plan_bytes) != baseline["evidence"]["plan_sha256"]):
        raise ValueError("The cases, rubric or local baseline plan differ from the published reference")
    plan = json.loads(plan_bytes)
    cases = [case for case in suite["cases"] if case["split"] == "development"]
    reserved = [case["case_id"] for case in suite["cases"] if case["split"] == "held_out"]
    if (len(cases) != 24 or cases != plan["cases"]
            or [case["case_id"] for case in cases] != [case["case_id"] for case in baseline["cases"]]
            or reserved != baseline["reserved_ids_not_executed"]):
        raise ValueError("Candidate cases must match the frozen development partition")

    fixtures = {label: path.replace("\\", "/") for label, path in plan["fixtures"].items()}
    source = "data/raw/nasa/package-1/B0005.mat"
    inputs = {**plan["fixture_hashes"], source: suite["source_identity"]["source_sha256"],
              BASELINE.relative_to(ROOT).as_posix(): sha(baseline_bytes),
              CASES.relative_to(ROOT).as_posix(): sha(case_bytes),
              rubric.relative_to(ROOT).as_posix(): sha(rubric_bytes),
              baseline_plan.relative_to(ROOT).as_posix(): sha(plan_bytes)}
    catalog = json.loads((ROOT / fixtures["available"]).read_bytes())
    inputs.update({doc["path"]: doc["sha256"] for doc in catalog["documents"]})
    missing_catalog = json.loads((ROOT / fixtures["missing"]).read_bytes())
    absent = [doc["path"] for doc in missing_catalog["documents"]]
    verify_inputs(inputs, absent)
    return baseline, suite, cases, fixtures, inputs, absent


def verify_inputs(hashes, absent):
    for path, expected in hashes.items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"Evaluation input changed: {path}")
    for path in absent:
        if (ROOT / path).exists():
            raise ValueError(f"A deliberately absent fixture exists: {path}")


def runtime_snapshot(output):
    names = set(git("ls-files").splitlines()) | set(git("ls-files", "--others", "--exclude-standard").splitlines())
    paths = sorted(name for name in names if name.startswith(("scripts/", "requirements", "data/manifests/")))
    hashes = {}
    for name in paths:
        data = (ROOT / name).read_bytes()
        destination = output / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[name] = sha(data)
    return hashes


def case_command(case, catalog, model, parent_id=None):
    command = [sys.executable, "-m", "scripts.run_agent_investigation",
               "--model", case.get("model_override", model), "--protocol-catalog", str(catalog)]
    if "resume_from_case" in case:
        if parent_id is None:
            raise ValueError("A resume case requires its candidate parent run ID")
        command.extend(("--resume", parent_id))
    else:
        command.extend(("--question", case["request"]["question"]))
    for field in SELECTION_FIELDS:
        value = case["request"][field]
        if value is not None:
            command.extend(("--" + field.replace("_", "-"), str(value)))
    return command


def execution_failures(case, suite, record, report, process_exit, *, parent=None):
    """Apply the frozen execution contract, including the new resume lineage."""
    if record is None:
        return ["required_run_record"]
    category = case["category"]
    expected_exit = 2 if category in REJECTION_REASONS else 0
    checks = {
        "process_exit": process_exit == expected_exit,
        "report_record_link": f"]({record['run_id']}.json)" in report,
        "no_pending_calls_or_raw_messages": record["pending_calls"] == [] and "messages" not in record,
        "attempts_within_limits": all(
            type(record[field]) is int and 0 <= record[field] <= suite["limits"][limit]
            for field, limit in (("model_attempts", "model_limit"), ("tool_attempts", "tool_limit"))
        ),
    }
    request = case["request"]
    if category == "resume":
        parent_id = parent["run_id"]
        checks["resume_lineage"] = (
            record.get("resumed_from_run_id") == parent_id != record["run_id"]
            and f"]({parent_id}.json)" in report
        )
        request = {**request, "question": parent["request"]["question"]}
        previous = parent["comparison"]
        revision = previous["selection"]["revision"] + 1 if previous else 1
        suite = {**suite, "source_identity": {**suite["source_identity"], "revision": revision}}
        old_trace = parent.get("trace_evidence")
        if old_trace:
            checks["previous_trace_excluded"] = old_trace["evidence_id"] not in json.dumps(record)
    checks["request_matches"] = same_json(record["request"], request)
    if category in {"out_of_range", "unavailable_model"}:
        checks["expected_failure"] = (record["status"] == "failed"
                                      and record["stop_reason"] == REJECTION_REASONS[category])
        checks["failure_detail"] = bool(record["trace"] and record["trace"][-1].get("detail"))
        checks["no_comparison_or_findings"] = record["comparison"] is None and record["findings"] is None
        if category == "unavailable_model":
            checks["no_model_or_tools_loaded"] = (
                record["configuration"]["model_used"] is None and record["tool_attempts"] == 0
            )
        return [name for name, passed in checks.items() if not passed]

    actions = case["allowed_actions"]
    if category in {"clarification", "invalid_request"}:
        actions = [None]
    elif category == "resume":
        actions = ["stop_with_partial", "inspect_discharge_traces"]
    protocol_status = {"available": "available", "partial": "partial"}.get(case["fixture"], "unavailable")
    alternatives = []
    for action in actions:
        trace = action == "inspect_discharge_traces"
        if action is None:
            status, reason = (("needs_clarification", "missing_selection") if category == "clarification"
                              else ("failed", "invalid_request"))
        else:
            status = "partial"
            reason = ("protocol_evidence_incomplete" if protocol_status != "available"
                      else "comparability_assessment_pending" if trace else "selected_trace_check_pending")
        expected = {"action": action, "trace_required": trace, "status": status,
                    "stop_reason": reason, "protocol_status": protocol_status}
        review = evaluate_follow_up(record, {**case, "request": request, "expected": expected},
                                    {**suite, "manual_review": ["Review source and task coverage separately."]})
        alternatives.append(review["failures"])
    decision = record.get("follow_up")
    if case["expected_scopes"] and decision:
        checks["expected_scope"] = decision.get("question_scope") in case["expected_scopes"]
    if record.get("findings") and record.get("protocol"):
        checks["availability_warning"] = (
            (record["findings"]["protocol_evidence_limitation"] is None)
            == (record["protocol"]["status"] == "available")
        )
    return sorted(set([name for name, passed in checks.items() if not passed] + min(alternatives, key=len)))


def execute(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=300, check=False)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        return "process_timeout", (error.stdout or b"").decode("utf-8", "replace"), (
            error.stderr or b"").decode("utf-8", "replace")
    except OSError as error:
        return "process_start_error", "", str(error)


def preload_model(expected):
    import httpx
    from scripts.local_model import LOCAL_URL, create_local_model
    from scripts.run_agent_investigation import MODEL_SETTINGS

    _, runtime = create_local_model(expected["model"], MODEL_SETTINGS)
    if runtime != expected:
        raise ValueError("The local model, server version or settings differ from the frozen baseline")
    started = perf_counter()
    with httpx.Client(base_url=LOCAL_URL, timeout=115.0, trust_env=False) as client:
        response = client.post("/api/chat", json={"model": runtime["model"], "stream": False, "keep_alive": "2m",
                                                 "options": {"num_ctx": MODEL_SETTINGS["num_ctx"]}})
        response.raise_for_status()
        if not response.json()["done"]:
            raise ValueError("The local model did not preload")
    return runtime, perf_counter() - started


def read_run(stdout, existing_ids):
    from scripts.investigation_resume import RUN_ID_PATTERN

    prefix = "Local run record: "
    path = next((Path(line[len(prefix):]).resolve() for line in reversed(stdout.splitlines()) if line.startswith(prefix)), None)
    if path is None:
        return None, None, ""
    if (path.parent != (ROOT / "outputs/agent-investigations").resolve() or path.suffix != ".json"
            or not RUN_ID_PATTERN.fullmatch(path.stem) or path.stem in existing_ids):
        raise ValueError("Unexpected investigation record path")
    record = json.loads(path.read_bytes())
    if record["run_id"] != path.stem:
        raise ValueError("Investigation run ID differs from the record path")
    report = path.with_suffix(".md")
    return path, record, report.read_text(encoding="utf-8") if report.is_file() else ""


def collect(output):
    if output.exists():
        raise ValueError("Use a fresh candidate directory to preserve earlier runs")
    baseline, suite, cases, fixtures, input_hashes, absent = read_inputs()
    output.mkdir(parents=True, exist_ok=False)
    runtime_hashes = runtime_snapshot(output)
    model, preload_seconds = preload_model(baseline["configuration"]["model_used"])
    result = {"checkout_commit": git("rev-parse", "HEAD"), "baseline_id": baseline["baseline_id"],
              "input_hashes": input_hashes, "absent_fixtures": absent, "runtime_hashes": runtime_hashes,
              "model_used": model, "preload_seconds": preload_seconds, "records": [],
              "status": "collecting", "runs_per_case": 1, "retries": 0,
              "reserved_ids_not_executed": baseline["reserved_ids_not_executed"],
              "review_status": "pending"}
    write_json(output / "collection.json", result)
    parents = {}
    records_directory = ROOT / "outputs/agent-investigations"
    for case in cases:
        verify_inputs({**input_hashes, **runtime_hashes}, absent)
        case_id = case["case_id"]
        parent_key = case.get("resume_from_case")
        parent_entry = parents.get(parent_key)
        parent = parent_entry[0] if parent_entry else None
        existing_ids = {path.stem for path in records_directory.glob("*.json")}
        print(f"Starting {case_id}", flush=True)
        started = perf_counter()
        if parent_key and (parent is None or sha(parent_entry[1].read_bytes()) != parent_entry[2]):
            process_exit, stdout, stderr = "blocked_parent", "", f"Missing successful candidate parent: {parent_key}"
        else:
            command = case_command(case, ROOT / fixtures[case["fixture"]], model["model"],
                                   parent["run_id"] if parent else None)
            process_exit, stdout, stderr = execute(command)
        wall = perf_counter() - started
        stdout_path, stderr_path = (output / f"{case_id}.{stream}.txt" for stream in ("stdout", "stderr"))
        stdout_path.write_text(stdout, encoding="utf-8", newline="\n")
        stderr_path.write_text(stderr, encoding="utf-8", newline="\n")
        path, record, report = None, None, ""
        try:
            path, record, report = read_run(stdout, existing_ids)
            failures = execution_failures(case, suite, record, report, process_exit, parent=parent)
        except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
            failures = [f"invalid_artifact:{type(error).__name__}"]
        if record and process_exit == 0 and not (set(failures) - {"expected_scope"}):
            parents[case_id] = (record, path, sha(path.read_bytes()))
        decision = (record or {}).get("follow_up") or {}
        usage = [event["usage"] for event in (record or {}).get("trace", []) if event.get("usage")]
        item = {"case_id": case_id, "process_exit": process_exit, "wall_seconds": wall,
                "stdout_sha256": sha(stdout_path.read_bytes()), "stderr_sha256": sha(stderr_path.read_bytes()),
                "run_id": path.stem if path else None, "record_path": path.relative_to(ROOT).as_posix() if path else None,
                "record_sha256": sha(path.read_bytes()) if path else None,
                "report_sha256": sha(path.with_suffix(".md").read_bytes()) if report else None,
                "status": (record or {}).get("status", "no_run_record"), "stop_reason": (record or {}).get("stop_reason"),
                "action": decision.get("action"), "question_scope": decision.get("question_scope"),
                "execution": int(not failures), "execution_failures": failures,
                "graph_seconds": (record or {}).get("elapsed_seconds"), "usage": usage}
        result["records"].append(item)
        write_json(output / "collection.json", result)
        print(f"Recorded {case_id}: {item['status']}, execution {'fail' if failures else 'pass'}", flush=True)
    verify_inputs({**input_hashes, **runtime_hashes}, absent)
    result["status"] = "collected"
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "collection.json", result)
    (output / "comparison.md").write_text(comparison_report(baseline, result), encoding="utf-8", newline="\n")


def comparison_report(baseline, result):
    original = {case["case_id"]: case for case in baseline["cases"]}
    records = result["records"]
    before = sum(case["scores"]["execution"] for case in original.values())
    after = sum(item["execution"] for item in records)
    timed = [item["wall_seconds"] for item in records if item["usage"]]
    lines = ["# Portfolio development execution comparison", "",
             f"Execution and declared scope: **{before}/{len(original)} baseline**, **{after}/{len(records)} candidate**.",
             "Numerical accuracy, citation support, task fulfilment and recommendation review are pending.", "",
             "| Case | Baseline execution | Candidate execution | Candidate checks needing review |",
             "| --- | --- | --- | --- |"]
    for item in records:
        case = item["case_id"]
        verdict = "Pass" if item["execution"] else "Fail"
        previous = "Pass" if original[case]["scores"]["execution"] else "Fail"
        lines.append(f"| {case} | {previous} | {verdict} | {', '.join(item['execution_failures']) or '—'} |")
    if timed:
        lines.extend(["", f"Candidate median CLI wall time was {median(timed):.2f} s across "
                           f"{len(timed)} cases with recorded model usage. Preload is measured separately."])
    lines.extend(["", "One attempt per development case. Six reserved variants are outside this collection.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory under outputs/")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Candidate output must stay under the project's outputs directory")
    try:
        collect(output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Candidate collection stopped: {error}\n")


if __name__ == "__main__":
    main()

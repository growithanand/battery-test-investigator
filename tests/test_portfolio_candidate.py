"""Check candidate collection boundaries without calling a model."""

import json
from pathlib import Path

import pytest

from scripts import collect_portfolio_candidate as candidate


def resume_case():
    return {"case_id": "resume", "category": "resume", "fixture": "available", "resume_from_case": "parent",
            "request": {"question": "Benchmark scenario text", "battery_id": "B0005", "early": 8, "late": 155}}


def test_resume_command_uses_candidate_parent_and_preserves_its_question():
    parent_id = "a" * 32
    command = candidate.case_command(resume_case(), Path("catalog.json"), "model", parent_id)
    assert command[command.index("--resume") + 1] == parent_id
    assert "--question" not in command and "Benchmark scenario text" not in command
    assert command[command.index("--early") + 1] == "8"
    assert command[command.index("--late") + 1] == "155"


def test_candidate_blocks_a_resume_with_no_successful_parent(tmp_path, monkeypatch):
    cases = [{"case_id": "parent", "category": "clarification", "fixture": "available",
              "request": {"question": "Compare", "battery_id": "B0005", "early": None, "late": 155}},
             resume_case()]
    baseline = {"baseline_id": "test-baseline", "configuration": {"model_used": {"model": "model"}},
                "reserved_ids_not_executed": ["reserved"], "cases": [
                    {"case_id": case["case_id"], "scores": {"execution": 0}} for case in cases
                ]}
    monkeypatch.setattr(candidate, "read_inputs", lambda: (baseline, {}, cases, {"available": "catalog.json"}, {}, []))
    monkeypatch.setattr(candidate, "runtime_snapshot", lambda output: {})
    monkeypatch.setattr(candidate, "git", lambda *args: "test-commit")
    monkeypatch.setattr(candidate, "preload_model", lambda model: (model, 0.0))
    commands = []

    def execute(command):
        commands.append(command)
        return 2, "", "constructed parent failure"

    monkeypatch.setattr(candidate, "execute", execute)
    output = tmp_path / "candidate"
    candidate.collect(output)
    collected = (output / "collection.json").read_bytes()
    results = json.loads(collected)["records"]
    assert len(commands) == 1 and "--resume" not in commands[0]
    assert results[1]["process_exit"] == "blocked_parent" and results[1]["execution"] == 0
    assert results[0]["process_exit"] == 2 and results[0]["execution"] == 0
    with pytest.raises(ValueError, match="fresh candidate directory"):
        candidate.collect(output)
    assert (output / "collection.json").read_bytes() == collected


def test_reported_run_must_be_a_fresh_record_under_the_project(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, "ROOT", tmp_path)
    directory = tmp_path / "outputs/agent-investigations"
    directory.mkdir(parents=True)
    path = directory / ("a" * 32 + ".json")
    path.write_text(json.dumps({"run_id": path.stem}), encoding="utf-8")
    stdout = f"Local run record: {path}"
    assert candidate.read_run(stdout, set())[0] == path
    with pytest.raises(ValueError, match="Unexpected investigation record path"):
        candidate.read_run(stdout, {path.stem})
    with pytest.raises(ValueError, match="Unexpected investigation record path"):
        candidate.read_run(f"Local run record: {tmp_path / path.name}", set())

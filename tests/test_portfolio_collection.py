"""Guard the frozen collection against an incorrect base or accidental rerun."""

import json
import sys

import pytest

from scripts import collect_portfolio_baseline as collector


def test_collection_output_must_stay_in_project_outputs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    monkeypatch.setattr(collector, "ROOT", project)
    monkeypatch.setattr(sys, "argv", ["collect_portfolio_baseline", "--prepare", "--output", str(outside)])
    with pytest.raises(SystemExit) as error:
        collector.main()
    assert error.value.code == 2
    assert not outside.exists()


def test_collection_rejects_changed_runtime_before_creating_fixtures(tmp_path, monkeypatch):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"baseline_commit": "reviewed-baseline"}), encoding="utf-8")
    monkeypatch.setattr(collector, "CASES", cases)

    def changed_runtime(*args):
        return "scripts/investigation_graph.py"

    monkeypatch.setattr(collector, "git", changed_runtime)
    with pytest.raises(AssertionError):
        collector.prepare(tmp_path)
    assert not (tmp_path / "plan.json").exists()
    assert not (tmp_path / "fixtures").exists()


def test_collection_accepts_new_documentation_when_baseline_runtime_matches(monkeypatch):
    commands = []

    def documentation_commit(*arguments):
        commands.append(arguments)
        if arguments[0] == "ls-tree":
            return "README.md\nscripts/investigation_graph.py\nrequirements-lock.txt"
        return ""

    monkeypatch.setattr(collector, "git", documentation_commit)
    collector.verify_baseline_runtime("frozen-baseline")
    assert commands[-1] == ("diff", "--name-only", "frozen-baseline", "--",
                            "scripts/investigation_graph.py", "requirements-lock.txt")


def test_existing_collection_is_preserved_without_model_preflight(tmp_path, monkeypatch):
    pytest.importorskip("langgraph", reason="Install the optional model dependency layer")
    from scripts import local_model

    existing = tmp_path / "collection.json"
    original = b'{"status":"collected","records":["retained-failure"]}\n'
    existing.write_bytes(original)

    def unexpected_model(*args):
        raise RuntimeError("A preserved collection must not start a new model preflight")

    monkeypatch.setattr(local_model, "create_local_model", unexpected_model)
    with pytest.raises(AssertionError, match="Collection exists"):
        collector.collect(tmp_path)
    assert existing.read_bytes() == original

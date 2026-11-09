"""Saved clarification validation before any investigation can start."""

import json

import pytest

from scripts.investigation_resume import resume_investigation


RUN_ID = "a" * 32


def test_resume_accepts_an_id_instead_of_an_artifact_path(tmp_path):
    with pytest.raises(ValueError, match="32-character run ID"):
        resume_investigation(tmp_path, f"../{RUN_ID}.json", battery_id=None, early=9, late=None)


@pytest.mark.parametrize("contents, message", [
    (None, "No saved investigation"),
    ('{"run_id":', "Could not read saved investigation"),
    (json.dumps({"run_id": "b" * 32}), "ID does not match"),
    (json.dumps({"run_id": RUN_ID, "status": "partial", "request": {
        "question": "Compare reported capacities", "battery_id": "B0005", "early": 1, "late": 2,
    }}), "requires a saved clarification or a run with a validated selection"),
])
def test_resume_rejects_unusable_saved_records(tmp_path, contents, message):
    if contents is not None:
        (tmp_path / f"{RUN_ID}.json").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        resume_investigation(tmp_path, RUN_ID, battery_id=None, early=9, late=None)


@pytest.mark.parametrize("changes", [{"early": True}, {"late": 3}, {"revision": 0}, {"revision": True}])
def test_revision_requires_a_consistent_saved_selection(tmp_path, changes):
    request = {"question": "Compare reported capacities", "battery_id": "B0005", "early": 1, "late": 2}
    selection = {
        "battery_id": "B0005", "early": 1, "late": 2, "early_operation_index": 2, "late_operation_index": 4,
        "source_sha256": "b" * 64, "revision": 1, **changes,
    }
    saved = {"run_id": RUN_ID, "status": "partial", "request": request, "comparison": {"selection": selection}}
    (tmp_path / f"{RUN_ID}.json").write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its request or has an invalid revision"):
        resume_investigation(tmp_path, RUN_ID, battery_id=None, early=None, late=3)

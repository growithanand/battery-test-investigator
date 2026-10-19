"""Constructed cases verify arithmetic, selections, and pipeline completion."""

import hashlib
import json

import numpy as np
import pytest
from scipy.io import savemat

from scripts import run_investigation as workflow
from scripts.inspect_sample import read_discharge_records


@pytest.fixture
def sample_file(tmp_path):
    operations = []
    for kind, capacity in [("charge", 2.0), ("discharge", 2.0), ("charge", 1.5), ("discharge", 1.5)]:
        operations.append({
            "type": kind,
            "ambient_temperature": 24,
            "data": {
                "Time": np.array([0.0, 1.0, 2.0]),
                "Voltage_measured": np.array([4.1, 4.0, 3.9]),
                "Current_measured": np.array([-2.0, -2.0, -2.0]),
                "Temperature_measured": np.array([24.0, 24.2, 24.4]),
                "Capacity": capacity,
            },
        })
    path = tmp_path / "BTEST.mat"
    savemat(path, {"BTEST": {"cycle": operations}})
    return path


def test_capacity_change_uses_earlier_capacity_and_preserves_source_indices(sample_file):
    _, records = read_discharge_records(sample_file)
    comparison = workflow.compare_discharges(records)
    assert comparison["capacity_change_Ah"] == pytest.approx(-0.5)
    assert comparison["capacity_change_percent"] == pytest.approx(-25.0)
    assert comparison["earlier"]["operation_index"] == 2
    assert comparison["later"]["operation_index"] == 4
    # Equal temperature/current summaries must not become an unsupported verdict.
    assert comparison["comparability"]["status"] == "not_established"


@pytest.mark.parametrize("early,late", [(0, None), (-1, None), (1, 3), (2, 1), (1, 1)])
def test_invalid_selections_fail_before_creating_outputs(sample_file, tmp_path, early, late):
    output_root = tmp_path / "outputs"
    with pytest.raises(ValueError, match="Choose discharge indices"):
        workflow.run_investigation(sample_file, output_root, early, late)
    assert not output_root.exists()


def test_repeat_run_reuses_folder_and_artifact_hashes_match(sample_file, tmp_path):
    output_root = tmp_path / "outputs"
    first_dir = workflow.run_investigation(sample_file, output_root)
    first_manifest = json.loads((first_dir / "complete.json").read_text(encoding="utf-8"))
    second_dir = workflow.run_investigation(sample_file, output_root, late=2)
    manifest = json.loads((second_dir / "complete.json").read_text(encoding="utf-8"))
    assert first_dir == second_dir
    assert len(list(output_root.iterdir())) == 1
    assert manifest == first_manifest
    assert manifest["status"] == "complete"
    assert manifest["source_sha256"] == hashlib.sha256(sample_file.read_bytes()).hexdigest()
    for name, digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((second_dir / name).read_bytes()).hexdigest() == digest
    comparison = json.loads((second_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["capacity_change_percent"] == pytest.approx(-25.0)
    report = (second_dir / "report.md").read_text(encoding="utf-8")
    assert "-25.000%" in report
    assert "not established" in report
    assert "This run made no external updates" in report


def test_failed_rerun_removes_stale_success_marker(sample_file, tmp_path, monkeypatch):
    output_root = tmp_path / "outputs"
    run_dir = workflow.run_investigation(sample_file, output_root)

    def fail_output(*args):
        raise OSError("Simulated output failure")

    monkeypatch.setattr(workflow, "write_inspection_outputs", fail_output)
    with pytest.raises(OSError, match="Simulated output failure"):
        workflow.run_investigation(sample_file, output_root)
    assert not (run_dir / "complete.json").exists()

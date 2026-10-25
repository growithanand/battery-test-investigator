"""Constructed traces verify descriptions and source identity, not battery diagnoses."""

from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest
from scipy.io import savemat

from scripts import investigation_tools
from scripts import trace_inspection as traces
from scripts.inspect_sample import read_discharge_records
from scripts.investigation_state import InvestigationRequest, resolve_selection


@pytest.fixture
def source(tmp_path):
    def operation(kind, time, current, voltage, temperature, capacity):
        return {"type": kind, "ambient_temperature": 24,
                "data": {"Time": np.array(time, dtype=float), "Current_measured": np.array(current, dtype=float),
                         "Voltage_measured": np.array(voltage, dtype=float),
                         "Temperature_measured": np.array(temperature, dtype=float), "Capacity": capacity}}

    first = ([10, 11, 15, 21, 24], [0, -2, -2, 0, 0], [4.2, 4.0, 2.6, 3.2, 3.3], [24, 25, 26, 25, 24], 2.0)
    later = ([100, 102, 106, 107, 112], [0, -1, -1, 0, 0], [4.2, 3.9, 2.7, 3.4, 3.5], [24, 26, 27, 25, 24], 1.5)
    constant = ([0, 1, 2, 3, 4], [-3] * 5, [4.1, 4.0, 3.5, 3.0, 2.8], [25, 26, 27, 28, 29], 1.0)
    operations = [operation("charge", *first), operation("discharge", *first), operation("impedance", *first),
                  operation("discharge", *later), operation("discharge", *constant)]
    path = tmp_path / "BTEST.mat"
    savemat(path, {"BTEST": {"cycle": operations}})
    return path


def selection(source, late=2, revision=1):
    return resolve_selection(InvestigationRequest("Inspect traces", "BTEST", 1, late), source, revision=revision)


def test_trace_evidence_preserves_selection_units_and_entire_record(source):
    chosen = selection(source)
    result = traces.inspect_selection_traces(chosen, source)
    assert result["selection"] == asdict(chosen)
    assert result["scope"] == "whole_recording" and result["statistics_basis"] == "unweighted_recorded_samples"
    earlier, later = result["earlier"], result["later"]
    assert (earlier["operation_index"], later["operation_index"]) == (2, 4)
    assert earlier["recording_duration_s"] == 14 and later["recording_duration_s"] == 12
    assert earlier["sample_statistics"]["current_A"] == {"minimum": -2.0, "median": 0.0, "maximum": 0.0}
    assert earlier["sample_statistics"]["temperature_C"]["maximum"] == 26.0
    assert earlier["first_sample"] == {"sample_index": 1, "source_time_s": 10.0, "elapsed_s": 0.0,
                                       "current_A": 0.0, "voltage_V": 4.2, "temperature_C": 24.0}
    assert later["first_sample"]["source_time_s"] == 100.0 and later["first_sample"]["elapsed_s"] == 0.0
    assert earlier["last_sample"]["sample_index"] == 5 and earlier["last_sample"]["voltage_V"] == 3.3
    assert earlier["minimum_voltage_sample"]["sample_index"] == 3
    assert earlier["minimum_voltage_sample"]["elapsed_s"] == 5.0
    assert earlier["minimum_voltage_sample"]["voltage_V"] == 2.6
    assert result["comparability"] == "not_established"
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert "measurements" not in earlier and "active_discharge_duration_s" not in earlier


def test_current_steps_keep_sample_pair_and_actual_sampling_interval(source):
    first = traces.inspect_selection_traces(selection(source), source)["earlier"]
    increase, decrease = first["largest_current_increase"], first["largest_current_decrease"]
    assert (increase["before"]["sample_index"], increase["after"]["sample_index"]) == (3, 4)
    assert increase["delta_current_A"] == 2.0 and increase["sample_interval_s"] == 6.0
    assert (decrease["before"]["sample_index"], decrease["after"]["sample_index"]) == (1, 2)
    assert decrease["delta_current_A"] == -2.0 and decrease["sample_interval_s"] == 1.0


def test_constant_current_has_no_increase_or_decrease(source):
    later = traces.inspect_selection_traces(selection(source, late=3), source)["later"]
    assert later["largest_current_increase"] is None and later["largest_current_decrease"] is None


def test_tied_extrema_and_steps_choose_first_source_occurrence(source):
    _, records = read_discharge_records(source)
    record = records[0]
    record["measurements"]["Current_measured"] = np.array([0, -2, 0, -2, 0], dtype=float)
    record["measurements"]["Voltage_measured"] = np.array([4.2, 2.6, 3.0, 2.6, 3.3])
    summary = traces.summarize_trace(record)
    assert summary["minimum_voltage_sample"]["sample_index"] == 2
    assert summary["largest_current_decrease"]["before"]["sample_index"] == 1
    assert summary["largest_current_increase"]["before"]["sample_index"] == 2


def test_analysis_does_not_change_source_arrays(source):
    _, records = read_discharge_records(source)
    original = {key: array.copy() for key, array in records[0]["measurements"].items()}
    traces.summarize_trace(records[0])
    for key, array in original.items():
        np.testing.assert_array_equal(records[0]["measurements"][key], array)


def test_different_pair_and_revision_have_their_own_evidence_identity(source):
    original_selection = selection(source)
    original = traces.inspect_selection_traces(original_selection, source)
    changed = traces.inspect_selection_traces(selection(source, late=3, revision=2), source)
    revised = traces.inspect_selection_traces(replace(original_selection, revision=2), source)
    assert original["evidence_id"] != changed["evidence_id"] != revised["evidence_id"]
    assert changed["later"]["sample_statistics"]["current_A"]["median"] == -3.0
    assert original == traces.inspect_selection_traces(original_selection, source)


def test_changed_source_is_rejected_before_trace_analysis(source, monkeypatch):
    chosen = selection(source)
    source.write_bytes(source.read_bytes() + b"changed")
    monkeypatch.setattr(traces, "summarize_trace", lambda *a: pytest.fail("No analysis of a changed source"))
    with pytest.raises(ValueError, match="Source hash does not match"):
        traces.inspect_selection_traces(chosen, source)


def test_mutation_during_read_is_rejected(source, monkeypatch):
    chosen = selection(source)
    original_reader = investigation_tools.read_discharge_records

    def read_then_change(path):
        result = original_reader(path)
        path.write_bytes(path.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(investigation_tools, "read_discharge_records", read_then_change)
    with pytest.raises(ValueError, match="Source changed while reading"):
        traces.inspect_selection_traces(chosen, source)


@pytest.mark.parametrize("changes", [{"early_operation_index": 1}, {"late_operation_index": 3}, {"late": 3}])
def test_wrong_operation_identity_cannot_label_trace_evidence(source, changes):
    with pytest.raises(ValueError, match="Operation indices do not match"):
        traces.inspect_selection_traces(replace(selection(source), **changes), source)


def test_out_of_bounds_selection_fails_before_creating_output(source, tmp_path):
    chosen = replace(selection(source), late=99)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Selected indices"):
        traces.write_trace_inspection(chosen, source, output)
    assert not output.exists()


def test_overflow_in_derived_current_steps_is_rejected(source):
    _, records = read_discharge_records(source)
    records[0]["measurements"]["Current_measured"] = np.array([-1e308, 1e308, 0, 0, 0])
    with pytest.raises(ValueError, match="must be finite"):
        traces.summarize_trace(records[0])


def test_plot_contains_every_selected_sample_at_elapsed_record_time(source, tmp_path, monkeypatch):
    from matplotlib import pyplot as plt

    chosen = selection(source)
    _, records = read_discharge_records(source)
    subplots = plt.subplots
    captured = {}

    def capture(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        captured.update(figure=figure, axes=axes)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture)
    path = tmp_path / "selected.png"
    traces.save_selected_trace_plot(records, chosen, path)
    for axis, field in zip(captured["axes"], ("Current_measured", "Voltage_measured", "Temperature_measured")):
        assert len(axis.lines) == 2
        for line, record in zip(axis.lines, records[:2]):
            arrays = record["measurements"]
            np.testing.assert_array_equal(line.get_xdata(), arrays["Time"] - arrays["Time"][0])
            np.testing.assert_array_equal(line.get_ydata(), arrays[field])
    assert path.read_bytes().startswith(b"\x89PNG")
    assert not plt.fignum_exists(captured["figure"].number)


def test_output_hash_and_repeat_identity_match_written_artifacts(source, tmp_path):
    chosen = selection(source)
    root = tmp_path / "outputs"
    directory = traces.write_trace_inspection(chosen, source, root)
    result = json.loads((directory / "trace-inspection.json").read_text(encoding="utf-8"))
    assert result["selection"] == asdict(chosen)
    assert result["artifacts"]["plot"]["sha256"] == hashlib.sha256((directory / "selected-traces.png").read_bytes()).hexdigest()
    assert directory == traces.write_trace_inspection(chosen, source, root)
    assert len(list(root.iterdir())) == 1


def test_failed_plot_rerun_cannot_leave_an_old_summary_as_success(source, tmp_path, monkeypatch):
    chosen = selection(source)
    root = tmp_path / "outputs"
    directory = traces.write_trace_inspection(chosen, source, root)

    def fail_plot(*args):
        raise OSError("Synthetic plot failure")

    monkeypatch.setattr(traces, "save_selected_trace_plot", fail_plot)
    with pytest.raises(OSError, match="Synthetic plot failure"):
        traces.write_trace_inspection(chosen, source, root)
    assert not (directory / "trace-inspection.json").exists()


def test_langchain_trace_tool_is_read_only_and_uses_same_evidence(source, tmp_path):
    pytest.importorskip("langchain_ollama", reason="Optional model layer")
    from scripts.model_tools import make_trace_tool

    before = set(tmp_path.rglob("*"))
    tool = make_trace_tool({"BTEST": source})
    result = tool.invoke({"battery_id": "BTEST", "early": 1, "late": 2})
    assert result == traces.inspect_selection_traces(selection(source), source)
    assert set(tmp_path.rglob("*")) == before
    assert set(tool.args_schema.model_fields) == {"battery_id", "early", "late"}


@pytest.mark.parametrize("changes", [{"source_path": "other.mat"}, {"output_dir": "elsewhere"},
                                     {"early": True}, {"late": "2"}, {"revision": 99}])
def test_langchain_trace_tool_rejects_paths_and_coercion(changes):
    pytest.importorskip("langchain_ollama", reason="Optional model layer")
    from pydantic import ValidationError
    from scripts.model_tools import make_trace_tool

    with pytest.raises(ValidationError):
        make_trace_tool({}).invoke({"battery_id": "BTEST", "early": 1, "late": 2, **changes})

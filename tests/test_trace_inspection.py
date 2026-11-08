"""Constructed traces verify descriptions and source identity, not battery diagnoses."""

from dataclasses import asdict, replace
import hashlib
import json
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat

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
    record["measurements"]["Temperature_measured"] = np.array([24, 24, 26, 26, 25], dtype=float)
    summary = traces.summarize_trace(record)
    assert summary["minimum_voltage_sample"]["sample_index"] == 2
    assert summary["largest_current_decrease"]["before"]["sample_index"] == 1
    assert summary["largest_current_increase"]["before"]["sample_index"] == 2
    assert {label: {kind: point["sample_index"] for kind, point in extrema.items()}
            for label, extrema in summary["extrema_samples"].items()} == {
                "current_A": {"minimum": 2, "maximum": 1},
                "voltage_V": {"minimum": 2, "maximum": 1},
                "temperature_C": {"minimum": 1, "maximum": 3},
            }


@pytest.mark.parametrize("label,kind,index", [
    ("current_A", "minimum", 1), ("current_A", "maximum", 2),
    ("voltage_V", "minimum", 3), ("voltage_V", "maximum", 4),
    ("temperature_C", "minimum", 5), ("temperature_C", "maximum", 6),
])
def test_every_signal_extremum_has_its_own_source_point(source, label, kind, index):
    _, records = read_discharge_records(source)
    record = records[0]
    record["sample_count"] = 7
    record["measurements"] = {
        "Time": np.array([12, 12.5, 15, 21, 23, 30, 31]),
        "Current_measured": np.array([-1, -4, 3, -2, 0, -3, 2]),
        "Voltage_measured": np.array([3.7, 3.8, 3.6, 2.5, 4.4, 3.9, 3.2]),
        "Temperature_measured": np.array([25, 26, 24, 27, 28, 19, 39]),
    }
    summary = traces.summarize_trace(record)
    point = summary["extrema_samples"][label][kind]
    assert point == {
        "sample_index": index + 1,
        "source_time_s": [12, 12.5, 15, 21, 23, 30, 31][index],
        "elapsed_s": [0, 0.5, 3, 9, 11, 18, 19][index],
        "current_A": [-1, -4, 3, -2, 0, -3, 2][index],
        "voltage_V": [3.7, 3.8, 3.6, 2.5, 4.4, 3.9, 3.2][index],
        "temperature_C": [25, 26, 24, 27, 28, 19, 39][index],
    }
    assert point[label] == summary["sample_statistics"][label][kind]
    assert summary["minimum_voltage_sample"] == summary["extrema_samples"]["voltage_V"]["minimum"]


@pytest.mark.parametrize("field,label,value", [
    ("Current_measured", "current_A", -2), ("Voltage_measured", "voltage_V", 3.5),
    ("Temperature_measured", "temperature_C", 24),
])
def test_constant_signal_extrema_both_locate_first_sample(source, field, label, value):
    _, records = read_discharge_records(source)
    records[0]["measurements"][field] = np.full(5, value, dtype=float)
    extrema = traces.summarize_trace(records[0])["extrema_samples"][label]
    for point in extrema.values():
        assert point["sample_index"] == 1 and point["source_time_s"] == 10 and point["elapsed_s"] == 0
        assert point[label] == value


def test_extended_evidence_has_a_new_analysis_identity(source):
    chosen = selection(source)
    result = traces.inspect_selection_traces(chosen, source)
    old_identity = {"analysis_version": "whole-record-traces-v1", "selection": asdict(chosen)}
    old_digest = hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()[:16]
    assert result["analysis_version"] == "whole-record-traces-v2"
    assert result["evidence_id"] != f"traces-{old_digest}"


def test_extrema_and_final_voltage_display_include_values_and_both_source_time_bases(source):
    evidence = traces.inspect_selection_traces(selection(source), source)
    text = traces.format_trace_extrema(evidence)
    assert evidence["evidence_id"] in text and evidence["selection"]["source_sha256"] in text
    assert "Earlier: discharge 1, operation 2" in text and "Later: discharge 2, operation 4" in text
    assert "first occurrence" in text and "12 significant digits" in text
    assert "active-load boundaries unresolved" in text and "comparability not established" in text
    assert "Minimum voltage is the lowest recorded reading; final voltage is the last recorded sample." in text
    assert "Neither location alone establishes the active-discharge cutoff or a physical cause." in text
    rows = [[cell.strip() for cell in line.split("|")] for line in text.splitlines()
            if any(f"| {kind} |" in line for kind in ("minimum", "maximum", "final"))]
    assert rows == [
        ["current (A)", "minimum", "-2", "2", "11", "1"],
        ["current (A)", "maximum", "0", "1", "10", "0"],
        ["voltage (V)", "minimum", "2.6", "3", "15", "5"],
        ["voltage (V)", "maximum", "4.2", "1", "10", "0"],
        ["voltage (V)", "final", "3.3", "5", "24", "14"],
        ["temperature (degC)", "minimum", "24", "1", "10", "0"],
        ["temperature (degC)", "maximum", "26", "3", "15", "5"],
        ["current (A)", "minimum", "-1", "2", "102", "2"],
        ["current (A)", "maximum", "0", "1", "100", "0"],
        ["voltage (V)", "minimum", "2.7", "3", "106", "6"],
        ["voltage (V)", "maximum", "4.2", "1", "100", "0"],
        ["voltage (V)", "final", "3.5", "5", "112", "12"],
        ["temperature (degC)", "minimum", "24", "1", "100", "0"],
        ["temperature (degC)", "maximum", "27", "3", "106", "6"],
    ]


def test_extrema_display_rejects_an_older_evidence_format(source):
    evidence = traces.inspect_selection_traces(selection(source), source)
    evidence["analysis_version"] = "whole-record-traces-v1"
    evidence["earlier"].pop("extrema_samples")
    with pytest.raises(ValueError, match="analysis version"):
        traces.format_trace_extrema(evidence)


def test_standalone_cli_displays_the_saved_extrema_table(source, tmp_path, monkeypatch, capsys):
    from scripts import inspect_selected_traces as runner

    root = tmp_path / "inspection"
    monkeypatch.setattr(sys, "argv", ["inspect_selected_traces", "--input", str(source),
                                     "--early", "1", "--late", "2", "--output-dir", str(root)])
    runner.main()
    output = capsys.readouterr().out
    saved = json.loads(next(root.glob("*/trace-inspection.json")).read_bytes())
    assert traces.format_trace_extrema(saved) in output
    assert output.count("| minimum |") == output.count("| maximum |") == 6


@pytest.mark.parametrize("current,reason,missing", [
    ([0, 2, 2, 0], "no_on_load_samples", None),
    ([0, -2, 0], "single_on_load_sample", None),
    ([0, -2, -2, 0, -2, -2, 0], "split_on_load_blocks", None),
    ([-2, -2, 0], "missing_transition_brackets", ["load_on"]),
    ([0, -2, -2], "missing_transition_brackets", ["load_off"]),
    ([-2, -2], "missing_transition_brackets", ["load_on", "load_off"]),
])
def test_active_load_leaves_ambiguous_windows_unresolved(current, reason, missing):
    result = traces.find_b0005_active_load_window(np.arange(len(current)), np.array(current))
    assert result["status"] == "unresolved" and result["reason"] == reason
    assert result.get("missing_brackets") == missing
    assert "observed_span_s" not in result and "load_on_bracket" not in result


@pytest.mark.parametrize("times,current,message", [
    ([0, 1], [0, -2, 0], "aligned"),
    ([0, 1, 1], [0, -2, 0], "strictly increasing"),
    ([0, 1, 2], [0, np.nan, 0], "samples must be finite"),
    ([-1e308, 0, 1e308], [0, -2, 0], "Elapsed sample times must be finite"),
])
def test_active_load_rejects_invalid_samples(times, current, message):
    with pytest.raises(ValueError, match=message):
        traces.find_b0005_active_load_window(np.array(times), np.array(current))


@pytest.mark.parametrize("registered", [True, False])
def test_active_load_cli_preserves_source_and_both_brackets(source, tmp_path, monkeypatch, capsys, registered):
    from scripts import inspect_selected_traces as runner

    b0005 = tmp_path / "B0005.mat"
    savemat(b0005, {"B0005": loadmat(source)["BTEST"]})
    digest = hashlib.sha256(b0005.read_bytes()).hexdigest()
    manifest = tmp_path / "data/manifests/B0005.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"files": [{"archive_member": "B0005.mat",
                                                "sha256": digest if registered else "0" * 64}]}), encoding="utf-8")
    monkeypatch.setattr(traces, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["inspect_selected_traces", "--input", str(b0005),
                                     "--early", "1", "--late", "2", "--active-load"])
    if not registered:
        with pytest.raises(SystemExit) as error:
            runner.main()
        captured = capsys.readouterr()
        assert error.value.code == 2 and captured.out == ""
        assert "does not match the registered manifest" in captured.err
        return

    runner.main()
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["current_threshold_A"] == -1.0
    assert evidence["selection"]["source_sha256"] == digest
    assert evidence["selection"]["early_operation_index"] == 2
    assert evidence["selection"]["late_operation_index"] == 4
    assert evidence["earlier"] == {
        "status": "observed",
        "on_load_sample_count": 2,
        "observed_span_s": 4.0,
        "load_on_bracket": {
            "before": {"sample_index": 1, "source_time_s": 10.0, "elapsed_s": 0.0, "current_A": 0.0},
            "after": {"sample_index": 2, "source_time_s": 11.0, "elapsed_s": 1.0, "current_A": -2.0},
        },
        "load_off_bracket": {
            "before": {"sample_index": 3, "source_time_s": 15.0, "elapsed_s": 5.0, "current_A": -2.0},
            "after": {"sample_index": 4, "source_time_s": 21.0, "elapsed_s": 11.0, "current_A": 0.0},
        },
    }
    assert evidence["later"]["load_on_bracket"]["after"] == {
        "sample_index": 2, "source_time_s": 102.0, "elapsed_s": 2.0, "current_A": -1.0,
    }
    assert evidence["later"]["load_off_bracket"]["after"]["source_time_s"] == 107.0


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

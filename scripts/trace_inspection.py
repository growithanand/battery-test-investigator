"""Whole-record trace summaries and a separate B0005 active-load check."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.inspect_sample import ROOT
from scripts.investigation_state import ValidatedSelection
from scripts.investigation_tools import read_selection_records


ANALYSIS_VERSION = "whole-record-traces-v2"
ACTIVE_LOAD_ANALYSIS_VERSION = "b0005-active-load-v1"
B0005_LOAD_THRESHOLD_A = -1.0


def find_b0005_active_load_window(time_s: np.ndarray, current_A: np.ndarray) -> dict:
    """Locate one bracketed block of samples under the trial B0005 current rule."""
    time_s = np.asarray(time_s, dtype=float)
    current_A = np.asarray(current_A, dtype=float)
    if time_s.ndim != 1 or time_s.shape != current_A.shape or time_s.size < 2:
        raise ValueError("Time and current must be aligned one-dimensional arrays with at least two samples")
    if not np.isfinite(time_s).all() or not np.isfinite(current_A).all():
        raise ValueError("Time and current samples must be finite")
    if np.any(time_s[1:] <= time_s[:-1]):
        raise ValueError("Sample times must be strictly increasing")
    with np.errstate(over="ignore"):
        elapsed_s = time_s - time_s[0]
    if not np.isfinite(elapsed_s).all():
        raise ValueError("Elapsed sample times must be finite")

    on_load = np.flatnonzero(current_A <= B0005_LOAD_THRESHOLD_A)
    if on_load.size < 2:
        reason = "no_on_load_samples" if not on_load.size else "single_on_load_sample"
        return {"status": "unresolved", "reason": reason}
    if np.any(np.diff(on_load) != 1):
        return {"status": "unresolved", "reason": "split_on_load_blocks"}

    first, last = int(on_load[0]), int(on_load[-1])
    missing_brackets = []
    if first == 0:
        missing_brackets.append("load_on")
    if last == time_s.size - 1:
        missing_brackets.append("load_off")
    if missing_brackets:
        return {
            "status": "unresolved",
            "reason": "missing_transition_brackets",
            "missing_brackets": missing_brackets,
        }

    def sample(index):
        return {
            "sample_index": index + 1,
            "source_time_s": float(time_s[index]),
            "elapsed_s": float(elapsed_s[index]),
            "current_A": float(current_A[index]),
        }

    return {
        "status": "observed",
        "on_load_sample_count": int(on_load.size),
        "observed_span_s": float(time_s[last] - time_s[first]),
        "load_on_bracket": {"before": sample(first - 1), "after": sample(first)},
        "load_off_bracket": {"before": sample(last), "after": sample(last + 1)},
    }


def inspect_b0005_active_load(selection: ValidatedSelection, source_path: Path) -> dict:
    """Apply the trial rule to both selected records in the registered source."""
    _require_registered_b0005(selection)
    return _build_active_load_evidence(selection, read_selection_records(selection, source_path))


def _require_registered_b0005(selection: ValidatedSelection) -> None:
    if selection.battery_id != "B0005":
        raise ValueError("The active-load rule requires the registered B0005 source")
    manifest = json.loads((ROOT / "data/manifests/B0005.json").read_text(encoding="utf-8"))
    entry = next((item for item in manifest["files"] if item["archive_member"] == "B0005.mat"), None)
    if entry is None or selection.source_sha256 != entry["sha256"]:
        raise ValueError("B0005 source does not match the registered manifest")


def _build_active_load_evidence(selection: ValidatedSelection, records: list[dict]) -> dict:
    result = {
        "analysis_version": ACTIVE_LOAD_ANALYSIS_VERSION,
        "current_threshold_A": B0005_LOAD_THRESHOLD_A,
        "selection": asdict(selection),
        "interpretation_limit": (
            "The brackets describe sampled transitions across the trial threshold. "
            "The observed span runs from the first to the last on-load sample. "
            "Test-condition comparability remains unresolved."
        ),
    }
    for side, discharge in (("earlier", selection.early), ("later", selection.late)):
        measurements = records[discharge - 1]["measurements"]
        result[side] = find_b0005_active_load_window(measurements["Time"], measurements["Current_measured"])
    return result


def summarize_trace(record: dict) -> dict:
    """Summarize validated arrays, preserving one-based source sample positions."""
    measurements = record["measurements"]
    time = measurements["Time"]
    current = measurements["Current_measured"]
    with np.errstate(over="ignore", invalid="ignore"):
        elapsed = time - time[0]
        current_steps = np.diff(current)
    if not np.isfinite(elapsed).all() or not np.isfinite(current_steps).all():
        raise ValueError("Trace elapsed times and current differences must be finite")

    def point(index):
        return {
            "sample_index": int(index + 1),
            "source_time_s": float(time[index]),
            "elapsed_s": float(elapsed[index]),
            "current_A": float(current[index]),
            "voltage_V": float(measurements["Voltage_measured"][index]),
            "temperature_C": float(measurements["Temperature_measured"][index]),
        }

    def step(index):
        return {
            "before": point(index), "after": point(index + 1),
            "delta_current_A": float(current_steps[index]),
            "sample_interval_s": float(time[index + 1] - time[index]),
        }

    increase, decrease = int(np.argmax(current_steps)), int(np.argmin(current_steps))
    statistics, extrema_samples = {}, {}
    for field, label in (("Current_measured", "current_A"), ("Voltage_measured", "voltage_V"),
                         ("Temperature_measured", "temperature_C")):
        values = measurements[field]
        with np.errstate(over="ignore", invalid="ignore"):
            statistics[label] = {"minimum": float(np.min(values)), "median": float(np.median(values)),
                                 "maximum": float(np.max(values))}
        if not all(np.isfinite(value) for value in statistics[label].values()):
            raise ValueError("Trace sample statistics must be finite")
        extrema_samples[label] = {
            "minimum": point(int(np.argmin(values))),
            "maximum": point(int(np.argmax(values))),
        }
    return {
        "operation_index": record["operation_index"], "discharge_index": record["discharge_index"],
        "sample_count": record["sample_count"], "recording_duration_s": float(elapsed[-1]),
        "reported_capacity_Ah": record["reported_capacity_Ah"],
        "sample_statistics": statistics,
        "extrema_samples": extrema_samples,
        "first_sample": point(0), "last_sample": point(len(time) - 1),
        "minimum_voltage_sample": extrema_samples["voltage_V"]["minimum"].copy(),
        "largest_current_increase": step(increase) if current_steps[increase] > 0 else None,
        "largest_current_decrease": step(decrease) if current_steps[decrease] < 0 else None,
    }


def _build_evidence(selection, records, active_load=None):
    identity = {"analysis_version": ANALYSIS_VERSION, "selection": asdict(selection)}
    if active_load is not None:
        identity["active_load_analysis_version"] = active_load["analysis_version"]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    evidence = {
        **identity, "evidence_id": f"traces-{digest}",
        "scope": "whole_recording", "statistics_basis": "unweighted_recorded_samples",
        "time_basis": "elapsed seconds since each record's first sample; no phase alignment",
        "earlier": summarize_trace(records[selection.early - 1]),
        "later": summarize_trace(records[selection.late - 1]),
        "comparability": "not_established",
        "limits": [
            "Recording duration covers the whole trace. The observed on-load span is reported separately."
            if active_load is not None else
            "No active-discharge segmentation: recording duration is not active-load duration.",
            "Signed current is retained as recorded; protocol current may describe load magnitude.",
            "Sample statistics are not time-weighted and include startup and any post-load readings.",
            "Extrema and adjacent current steps are descriptive, not cutoff or load-state classifiers; ties use the first occurrence.",
            "No current integration, protocol pass/fail tolerance, physical cause or end-of-life verdict is inferred.",
        ],
    }
    if active_load is not None:
        evidence["active_load"] = active_load
    return evidence


def inspect_selection_traces(selection: ValidatedSelection, source_path: Path, *, include_active_load: bool = False) -> dict:
    """Return compact selection-bound trace evidence; keep raw arrays outside model context."""
    if include_active_load:
        _require_registered_b0005(selection)
    records = read_selection_records(selection, source_path)
    active_load = _build_active_load_evidence(selection, records) if include_active_load else None
    return _build_evidence(selection, records, active_load)


def _format_active_load_windows(evidence: dict) -> list[str]:
    lines = [
        f"Observed active-load windows ({evidence['analysis_version']})",
        f"On-load rule: signed current <= {evidence['current_threshold_A']:g} A.",
        evidence["interpretation_limit"],
    ]
    selection = evidence["selection"]
    for side, index in (("earlier", "early"), ("later", "late")):
        window = evidence[side]
        lines.append(f"{side.capitalize()}: discharge {selection[index]}, operation {selection[f'{index}_operation_index']}")
        if window["status"] == "unresolved":
            reason = window["reason"].replace("_", " ")
            missing = window.get("missing_brackets", [])
            if missing:
                reason += " (" + ", ".join(label.replace("_", "-") for label in missing) + ")"
            lines.append(f"  Window unresolved: {reason}.")
            continue
        load_on, load_off = window["load_on_bracket"], window["load_off_bracket"]
        lines.append(
            f"  On-load samples {load_on['after']['sample_index']} to {load_off['before']['sample_index']} "
            f"({window['on_load_sample_count']} samples); observed span {window['observed_span_s']:.12g} s."
        )
        for label, bracket in (("Load-on", load_on), ("Load-off", load_off)):
            lines.append(f"  {label} bracket:")
            for position in ("before", "after"):
                point = bracket[position]
                lines.append(
                    f"    {position.capitalize()} sample {point['sample_index']}: "
                    f"source {point['source_time_s']:.12g} s, elapsed {point['elapsed_s']:.12g} s, "
                    f"current {point['current_A']:.12g} A."
                )
    return lines


def format_trace_extrema(evidence: dict) -> str:
    """Render located extrema and final voltage from the current recorded evidence."""
    if evidence["analysis_version"] != ANALYSIS_VERSION:
        raise ValueError("Extrema display requires the current trace analysis version")
    selection = evidence["selection"]
    active_load = evidence.get("active_load")
    lines = [
        f"Trace extrema and final voltage: {evidence['evidence_id']} ({evidence['analysis_version']})",
        f"Battery: {selection['battery_id']}; source SHA256: {selection['source_sha256']}",
        "Scope: whole recording; comparability not established. See the separate load-window check below."
        if active_load is not None else
        "Scope: whole recording; active-load boundaries unresolved; comparability not established.",
        "Ties use the first occurrence. Display rounded to 12 significant digits; JSON retains stored floats.",
        "Minimum voltage is the lowest recorded reading; final voltage is the last recorded sample.",
        "Neither location alone establishes the active-discharge cutoff or a physical cause.",
    ]
    for side in ("earlier", "later"):
        record = evidence[side]
        lines.extend([
            f"{side.capitalize()}: discharge {record['discharge_index']}, operation {record['operation_index']}",
            f"{'Signal (unit)':<18} | Reading | {'Value':>20} | {'Sample':>6} | "
            f"{'Source time (s)':>20} | {'Elapsed (s)':>20}",
        ])
        for label, name in (("current_A", "current (A)"), ("voltage_V", "voltage (V)"),
                            ("temperature_C", "temperature (degC)")):
            kinds = ("minimum", "maximum", "final") if label == "voltage_V" else ("minimum", "maximum")
            for kind in kinds:
                point = record["last_sample"] if kind == "final" else record["extrema_samples"][label][kind]
                lines.append(f"{name:<18} | {kind} | {point[label]:>20.12g} | {point['sample_index']:>6} | "
                             f"{point['source_time_s']:>20.12g} | {point['elapsed_s']:>20.12g}")
    if active_load is not None:
        lines.extend(["", *_format_active_load_windows(active_load)])
    return "\n".join(lines)


def save_selected_trace_plot(records: list[dict], selection: ValidatedSelection, destination: Path):
    """Plot every recorded sample for the selected pair, without resampling or clipping."""
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained")
    try:
        for index in (selection.early, selection.late):
            record = records[index - 1]
            measurements = record["measurements"]
            elapsed = measurements["Time"] - measurements["Time"][0]
            label = f"Discharge {index} (operation {record['operation_index']})"
            for axis, field in zip(axes, ("Current_measured", "Voltage_measured", "Temperature_measured")):
                axis.plot(elapsed, measurements[field], label=label)
        for axis, label in zip(axes, ("Recorded current (A)", "Voltage (V)", "Battery temperature (°C)")):
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        axes[0].legend()
        axes[-1].set(xlabel="Elapsed time since each record's first sample (s)", xlim=(0, None))
        figure.suptitle(f"{selection.battery_id}: selected whole-record traces\n"
                        "Signed current retained; no active window or phase alignment")
        figure.savefig(destination, dpi=140)
    finally:
        plt.close(figure)


def write_trace_inspection(selection: ValidatedSelection, source_path: Path, output_root: Path) -> Path:
    """Write artifacts from one validated snapshot; this is not agent completion."""
    records = read_selection_records(selection, source_path)
    evidence = _build_evidence(selection, records)
    destination = Path(output_root) / f"{selection.battery_id}-{evidence['evidence_id']}"
    destination.mkdir(parents=True, exist_ok=True)
    summary_path, plot_path = destination / "trace-inspection.json", destination / "selected-traces.png"
    # A failed rerun must not leave an earlier summary appearing newly successful.
    summary_path.unlink(missing_ok=True)
    save_selected_trace_plot(records, selection, plot_path)
    evidence["artifacts"] = {"plot": {"filename": plot_path.name,
                                     "sha256": hashlib.sha256(plot_path.read_bytes()).hexdigest()}}
    summary_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination

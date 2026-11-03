"""Descriptive whole-record trace evidence, without load segmentation or verdicts."""

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


def _build_evidence(selection, records):
    identity = {"analysis_version": ANALYSIS_VERSION, "selection": asdict(selection)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {
        **identity, "evidence_id": f"traces-{digest}",
        "scope": "whole_recording", "statistics_basis": "unweighted_recorded_samples",
        "time_basis": "elapsed seconds since each record's first sample; no phase alignment",
        "earlier": summarize_trace(records[selection.early - 1]),
        "later": summarize_trace(records[selection.late - 1]),
        "comparability": "not_established",
        "limits": [
            "No active-discharge segmentation: recording duration is not active-load duration.",
            "Signed current is retained as recorded; protocol current may describe load magnitude.",
            "Sample statistics are not time-weighted and include startup and any post-load readings.",
            "Extrema and adjacent current steps are descriptive, not cutoff or load-state classifiers; ties use the first occurrence.",
            "No current integration, protocol pass/fail tolerance, physical cause or end-of-life verdict is inferred.",
        ],
    }


def inspect_selection_traces(selection: ValidatedSelection, source_path: Path) -> dict:
    """Return compact selection-bound trace evidence; keep raw arrays outside model context."""
    return _build_evidence(selection, read_selection_records(selection, source_path))


def format_trace_extrema(evidence: dict) -> str:
    """Render the located extrema returned by the current analysis, without inference."""
    if evidence["analysis_version"] != ANALYSIS_VERSION:
        raise ValueError("Extrema display requires the current trace analysis version")
    selection = evidence["selection"]
    lines = [
        f"Trace extrema: {evidence['evidence_id']} ({evidence['analysis_version']})",
        f"Battery: {selection['battery_id']}; source SHA256: {selection['source_sha256']}",
        "Scope: whole recording; active-load boundaries unresolved; comparability not established.",
        "Ties use the first occurrence. Display rounded to 12 significant digits; JSON retains stored floats.",
    ]
    for side in ("earlier", "later"):
        record = evidence[side]
        lines.extend([
            f"{side.capitalize()}: discharge {record['discharge_index']}, operation {record['operation_index']}",
            f"{'Signal (unit)':<18} | Extremum | {'Value':>20} | {'Sample':>6} | "
            f"{'Source time (s)':>20} | {'Elapsed (s)':>20}",
        ])
        for label, name in (("current_A", "current (A)"), ("voltage_V", "voltage (V)"),
                            ("temperature_C", "temperature (degC)")):
            for kind in ("minimum", "maximum"):
                point = record["extrema_samples"][label][kind]
                lines.append(f"{name:<18} | {kind} | {point[label]:>20.12g} | {point['sample_index']:>6} | "
                             f"{point['source_time_s']:>20.12g} | {point['elapsed_s']:>20.12g}")
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

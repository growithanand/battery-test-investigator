"""Run a reproducible discharge comparison and write a local evidence report."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.inspect_sample import CSV_FIELDS, ROOT, read_discharge_records, write_inspection_outputs

WORKFLOW_VERSION = "reference-comparison-v1"


def compare_discharges(records, early=1, late=None):
    """Compare source-reported capacity; do not infer cause or comparability."""
    late = len(records) if late is None else late
    if not 1 <= early < late <= len(records):
        raise ValueError(
            f"Choose discharge indices with 1 <= early < late <= {len(records)}; "
            f"received early={early}, late={late}"
        )
    earlier = {key: records[early - 1][key] for key in CSV_FIELDS}
    later = {key: records[late - 1][key] for key in CSV_FIELDS}
    change = later["reported_capacity_Ah"] - earlier["reported_capacity_Ah"]
    return {
        "earlier": earlier,
        "later": later,
        "capacity_change_Ah": change,
        "capacity_change_percent": change / earlier["reported_capacity_Ah"] * 100,
        "percentage_reference": "earlier selected discharge's reported capacity",
        "ambient_temperature_change_C": later["ambient_temperature_C"] - earlier["ambient_temperature_C"],
        "median_current_change_A": later["median_current_A"] - earlier["median_current_A"],
        "comparability": {
            "status": "not_established",
            "reason": "Protocol verification and detailed load-profile matching are not implemented. "
                      "Ambient-temperature metadata and median current alone do not establish comparable conditions.",
        },
    }


def render_report(summary, comparison, investigation_id):
    """A fixed report is the baseline for the future model-assisted report."""
    earlier, later = comparison["earlier"], comparison["later"]
    cell_id = summary["cell_id"]
    table = [
        ("Discharge index", "discharge_index", "d"),
        ("Original operation index", "operation_index", "d"),
        ("Reported capacity (Ah)", "reported_capacity_Ah", ".6f"),
        ("Recorded ambient temperature (C)", "ambient_temperature_C", ".2f"),
        ("Median measured current (A)", "median_current_A", ".6f"),
        ("Maximum measured battery temperature (C)", "maximum_temperature_C", ".3f"),
        ("Recording span (s)", "recording_duration_s", ".3f"),
    ]
    rows = "\n".join(
        f"| {label} | {earlier[key]:{format_spec}} | {later[key]:{format_spec}} |"
        for label, key, format_spec in table
    )
    return f"""# {cell_id}: discharge investigation

Investigation ID: `{investigation_id}`. Workflow: `{WORKFLOW_VERSION}`.

This is an automated numerical reference report. No language model or document retrieval was used.

## Question and selection

Compare discharge {earlier['discharge_index']} with discharge {later['discharge_index']} in `{summary['source_filename']}`.
The file contains {sum(summary['operation_counts'].values())} operations, including {summary['discharge_operations']} discharges.

| Measurement | Earlier selection | Later selection |
| --- | ---: | ---: |
{rows}

## Calculated observation

Reported capacity changed by **{comparison['capacity_change_Ah']:+.6f} Ah**,
or **{comparison['capacity_change_percent']:+.3f}%** relative to the earlier selected discharge.

Formula: `(later reported capacity - earlier reported capacity) / earlier reported capacity * 100`.
These capacities are values supplied in the source file, not independently integrated or predicted values.

## Conditions and interpretation limits

- Comparability: **not established**. {comparison['comparability']['reason']}
- Recorded ambient-temperature difference: {comparison['ambient_temperature_change_C']:+.3f} C.
- Median measured-current difference: {comparison['median_current_change_A']:+.6f} A. Similar medians do not prove identical current profiles.
- The recording span can include readings after the main discharge load; it is not a segmented active-discharge duration.
- No physical degradation mechanism, remaining life, or end-of-life classification is inferred.
- The source-file hash establishes which file was analyzed; it does not replace experiment-document citations.

## Follow-up

Verify the applicable experiment protocol and inspect the selected current/voltage traces before making a stronger engineering interpretation.
Jira and Confluence publishing are not implemented. This run made no external updates.

## Artifacts and provenance

- [Inspection plot]({cell_id}-inspection.png): full capacity trend and the **first source discharge's** voltage/temperature traces, not an overlay of the selected pair.
- [All discharge summaries]({cell_id}-discharge-summary.csv)
- [Inspection metadata]({cell_id}-inspection.json)
- [Selected comparison and unrounded values](comparison.json)
- [Completed-run manifest](complete.json)

Source SHA-256: `{summary['source_sha256']}`.

For the NASA reference file, source attribution and download provenance are recorded in the repository's `data/manifests/B0005.json`.
"""


def run_investigation(input_path, output_root, early=1, late=None):
    """Validate before writing; identical source and selections reuse one folder."""
    input_path, output_root = Path(input_path), Path(output_root)
    counts, records = read_discharge_records(input_path)
    comparison = compare_discharges(records, early, late)
    identity = {
        "workflow_version": WORKFLOW_VERSION,
        "cell_id": input_path.stem,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "early": comparison["earlier"]["discharge_index"],
        "late": comparison["later"]["discharge_index"],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    investigation_id = f"{input_path.stem}-{digest}"
    run_dir = output_root / investigation_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not retain an earlier run's success marker.
    completion_path = run_dir / "complete.json"
    completion_path.unlink(missing_ok=True)
    summary = write_inspection_outputs(input_path, run_dir, counts, records)
    comparison.update({"investigation_id": investigation_id, "source_sha256": summary["source_sha256"]})
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(summary, comparison, investigation_id), encoding="utf-8")
    artifact_names = [f"{input_path.stem}-discharge-summary.csv", f"{input_path.stem}-inspection.json",
                      f"{input_path.stem}-inspection.png", "comparison.json", "report.md"]
    manifest = {
        **identity,
        "investigation_id": investigation_id,
        "status": "complete",
        "artifacts_sha256": {
            name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest() for name in artifact_names
        },
    }
    completion_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/nasa/package-1/B0005.mat")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/investigations")
    parser.add_argument("--early", type=int, default=1, help="Earlier discharge index, starting at 1")
    parser.add_argument("--late", type=int, help="Later discharge index; defaults to the final discharge")
    args = parser.parse_args()
    try:
        run_dir = run_investigation(args.input, args.output_dir, args.early, args.late)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Investigation failed: {error}\n")
    print(f"Report: {run_dir / 'report.md'}")
    print(f"Completed-run manifest: {run_dir / 'complete.json'}")


if __name__ == "__main__":
    main()

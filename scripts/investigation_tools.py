"""Deterministic analysis operations for the investigation workflow."""

from dataclasses import asdict
import hashlib
from pathlib import Path

from scripts.inspect_sample import read_discharge_records
from scripts.investigation_state import ValidatedSelection
from scripts.run_investigation import compare_discharges


def read_selection_records(selection: ValidatedSelection, source_path: Path) -> list[dict]:
    """Recheck source/selection identity before returning full analysis-layer records."""
    source_path = Path(source_path)
    if source_path.stem != selection.battery_id:
        raise ValueError(
            f"Selected battery {selection.battery_id} does not match source {source_path.stem}"
        )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != selection.source_sha256:
        raise ValueError("Source hash does not match the validated selection; resolve it again")

    _, records = read_discharge_records(source_path)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("Source changed while reading selected measurements; resolve it again")

    if not 1 <= selection.early < selection.late <= len(records):
        raise ValueError(f"Selected indices must satisfy 1 <= early < late <= {len(records)}")
    if (
        records[selection.early - 1]["operation_index"] != selection.early_operation_index
        or records[selection.late - 1]["operation_index"] != selection.late_operation_index
    ):
        raise ValueError("Operation indices do not match the validated selection; resolve it again")
    return records


def compare_selection(selection: ValidatedSelection, source_path: Path) -> dict:
    """Return compact numerical evidence without file-writing or model calls."""
    records = read_selection_records(selection, source_path)
    comparison = compare_discharges(records, selection.early, selection.late)

    return {
        "selection": asdict(selection),
        "comparison": comparison,
    }

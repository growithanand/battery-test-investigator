"""Deterministic analysis operations for the investigation workflow."""

from dataclasses import asdict
import hashlib
from pathlib import Path

from scripts.inspect_sample import read_discharge_records
from scripts.investigation_state import ValidatedSelection
from scripts.run_investigation import compare_discharges


def compare_selection(selection: ValidatedSelection, source_path: Path) -> dict:
    """Compare records belonging to an identity produced by resolve_selection.

    The caller supplies the trusted source path. Return compact, serializable
    evidence for the future agent; do not write reports or completion markers.
    """
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
        raise ValueError("Source changed while reading comparison measurements; resolve it again")

    comparison = compare_discharges(records, selection.early, selection.late)
    if (
        comparison["earlier"]["operation_index"] != selection.early_operation_index
        or comparison["later"]["operation_index"] != selection.late_operation_index
    ):
        raise ValueError("Operation indices do not match the validated selection; resolve it again")

    return {
        "selection": asdict(selection),
        "comparison": comparison,
    }

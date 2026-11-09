"""Recover the request and selection identity needed to continue a saved run."""

import json
from pathlib import Path
import re

from scripts.investigation_state import InvestigationRequest, ValidatedSelection


RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def _read_saved_run(directory: Path, run_id: str) -> dict:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("Resume requires a 32-character run ID from the saved investigation")
    path = directory / f"{run_id}.json"
    try:
        resolved = path.resolve(strict=True)
        if resolved != directory.resolve() / path.name or not resolved.is_file():
            raise ValueError("Expected a saved run file in the investigation directory")
        saved = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"No saved investigation for run {run_id}") from None
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError(f"Could not read saved investigation {run_id}") from error
    if not isinstance(saved, dict) or saved.get("run_id") != run_id:
        raise ValueError("Saved investigation ID does not match the requested run")
    return saved


def resume_investigation(
    directory: Path, run_id: str, *, battery_id: str | None, early: int | None, late: int | None,
) -> tuple[InvestigationRequest, ValidatedSelection | None]:
    """Fill a saved clarification or change a pair with an existing selection identity."""
    saved = _read_saved_run(directory, run_id)
    try:
        original = InvestigationRequest(**saved["request"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved investigation has an invalid request") from error
    clarification = saved.get("status") == "needs_clarification" and saved.get("stop_reason") == "missing_selection"
    previous = None
    if clarification:
        if not original.missing_fields():
            raise ValueError("Saved clarification has no missing selection fields")
    else:
        comparison = saved.get("comparison")
        if saved.get("status") not in ("partial", "failed", "complete") or not isinstance(comparison, dict):
            raise ValueError("Resume requires a saved clarification or a run with a validated selection")
        try:
            previous = ValidatedSelection(**comparison["selection"])
        except (KeyError, TypeError) as error:
            raise ValueError("Saved comparison has an invalid selection identity") from error
        if (original.missing_fields() or previous.battery_id != original.battery_id
                or type(previous.early) is not int or previous.early != original.early
                or type(previous.late) is not int or previous.late != original.late
                or type(previous.revision) is not int or previous.revision < 1):
            raise ValueError("Saved comparison selection does not match its request or has an invalid revision")

    selection = {"battery_id": battery_id, "early": early, "late": late}
    for field, supplied in selection.items():
        prior = getattr(original, field)
        if prior is None:
            continue
        if supplied is None:
            selection[field] = prior
        elif (clarification or field == "battery_id") and (type(supplied) is not type(prior) or supplied != prior):
            raise ValueError(f"Resume cannot change the saved {field} selection")
    completed = InvestigationRequest(question=original.question, **selection)
    if completed.missing_fields():
        flags = ", ".join("--" + field.replace("_", "-") for field in completed.missing_fields())
        raise ValueError(f"Complete the saved selection by supplying {flags}")
    if previous is not None and completed == original:
        raise ValueError("Supply a different discharge pair to revise this saved investigation")
    return completed, previous

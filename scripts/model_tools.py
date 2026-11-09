"""Narrow model-facing adapters around deterministic investigation tools."""

from collections.abc import Mapping
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from scripts.investigation_state import InvestigationRequest, ValidatedSelection, resolve_selection
from scripts.investigation_tools import compare_selection
from scripts.protocol_evidence import CATALOG_PATH, ROOT, retrieve_protocol_evidence
from scripts.trace_inspection import inspect_selection_traces


class ComparisonArguments(BaseModel):
    """Reject coercion and unexpected inputs before accessing source files."""

    model_config = ConfigDict(strict=True, extra="forbid")

    battery_id: str = Field(min_length=1, description="Registered battery ID, e.g. B0005")
    early: int = Field(gt=0, description="Earlier discharge-only index, starting at 1")
    late: int = Field(gt=0, description="Later discharge-only index, greater than early")


def _tool_selection(
    request: InvestigationRequest, source_path: Path, validated_selection: ValidatedSelection | None,
) -> ValidatedSelection:
    if validated_selection is None:
        return resolve_selection(request, source_path)
    requested = (request.battery_id, request.early, request.late)
    expected = (validated_selection.battery_id, validated_selection.early, validated_selection.late)
    if requested != expected:
        raise ValueError("Tool arguments differ from the validated selection")
    return validated_selection


def make_comparison_tool(
    source_paths: Mapping[str, Path], *, validated_selection: ValidatedSelection | None = None,
) -> BaseTool:
    """Keep the source registry in Python, outside the model-visible arguments."""
    sources = {battery_id: Path(path) for battery_id, path in source_paths.items()}

    @tool("compare_discharge_capacity", args_schema=ComparisonArguments)
    def compare_discharge_capacity(battery_id: str, early: int, late: int) -> dict:
        """Compare supplied discharge capacities for an explicitly selected pair.

        Only registered batteries are available. Indices refer to discharges,
        not all source operations. Return measured changes and source identity;
        this does not establish comparable protocols or a physical cause.
        """
        request = InvestigationRequest("Compare supplied capacities", battery_id, early, late)
        if battery_id not in sources:
            raise ValueError(f"Battery is not registered: {battery_id}")
        source_path = sources[battery_id]
        selection = _tool_selection(request, source_path, validated_selection)
        return compare_selection(selection, source_path)

    return compare_discharge_capacity


class ProtocolArguments(BaseModel):
    """Only battery identity is model-controlled; source configuration is not."""

    model_config = ConfigDict(strict=True, extra="forbid")

    battery_id: str = Field(pattern=r"^B[0-9]{4}$", description="Battery whose experiment protocol is needed")


def make_protocol_tool(catalog_path: Path = CATALOG_PATH, *, root: Path = ROOT) -> BaseTool:
    @tool("get_battery_protocol", args_schema=ProtocolArguments)
    def get_battery_protocol(battery_id: str) -> dict:
        """Retrieve applicable registered experiment passages with hashes and line citations.

        Unavailable or invalid sources return explicit issues, never substitute
        another battery's protocol. These are documentation claims, not proof
        of actual conditions, pair comparability, or a physical cause.
        """
        return retrieve_protocol_evidence(battery_id, catalog_path, root=root)

    return get_battery_protocol


def make_trace_tool(
    source_paths: Mapping[str, Path], *, include_active_load: bool = False,
    validated_selection: ValidatedSelection | None = None,
) -> BaseTool:
    """Expose read-only trace analysis, not artifact paths or file-writing authority."""
    sources = {battery_id: Path(path) for battery_id, path in source_paths.items()}
    active_load_battery = "B0005" if include_active_load and "B0005" in sources else None

    @tool("inspect_discharge_traces", args_schema=ComparisonArguments)
    def inspect_discharge_traces(battery_id: str, early: int, late: int) -> dict:
        """Inspect signed current/voltage/temperature for the explicit discharge pair.

        Return whole-record statistics and located extrema/current steps. When configured
        for registered B0005, also check sampled load windows. The tool writes no files.
        """
        request = InvestigationRequest("Inspect selected traces", battery_id, early, late)
        if battery_id not in sources:
            raise ValueError(f"Battery is not registered: {battery_id}")
        source_path = sources[battery_id]
        return inspect_selection_traces(
            _tool_selection(request, source_path, validated_selection), source_path,
            include_active_load=battery_id == active_load_battery,
        )

    inspect_discharge_traces.metadata = {"active_load_battery": active_load_battery}
    return inspect_discharge_traces

"""Narrow model-facing adapters around deterministic investigation tools."""

from collections.abc import Mapping
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from scripts.investigation_state import InvestigationRequest, resolve_selection
from scripts.investigation_tools import compare_selection
from scripts.protocol_evidence import CATALOG_PATH, ROOT, retrieve_protocol_evidence


class ComparisonArguments(BaseModel):
    """Reject coercion and unexpected inputs before accessing source files."""

    model_config = ConfigDict(strict=True, extra="forbid")

    battery_id: str = Field(min_length=1, description="Registered battery ID, e.g. B0005")
    early: int = Field(gt=0, description="Earlier discharge-only index, starting at 1")
    late: int = Field(gt=0, description="Later discharge-only index, greater than early")


def make_comparison_tool(source_paths: Mapping[str, Path]) -> BaseTool:
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
        selection = resolve_selection(request, source_path)
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

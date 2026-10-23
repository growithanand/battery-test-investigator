"""Constructed sources verify comparison evidence and stale-source rejection."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from scipy.io import savemat

from scripts.investigation_state import InvestigationRequest, resolve_selection
from scripts import investigation_tools as tools


@pytest.fixture
def sample_source(tmp_path):
    operations = []
    for kind, capacity in [
        ("charge", 2.0), ("discharge", 2.0), ("impedance", 2.0),
        ("discharge", 1.8), ("charge", 1.8), ("discharge", 1.5),
    ]:
        operations.append({
            "type": kind,
            "ambient_temperature": 24,
            "data": {
                "Time": np.array([0.0, 1.0, 2.0]),
                "Voltage_measured": np.array([4.1, 4.0, 3.9]),
                "Current_measured": np.array([-2.0, -2.0, -2.0]),
                "Temperature_measured": np.array([24.0, 24.2, 24.4]),
                "Capacity": capacity,
            },
        })
    path = tmp_path / "BTEST.mat"
    savemat(path, {"BTEST": {"cycle": operations}})
    return path


def test_comparison_preserves_selection_identity_and_calculation_basis(sample_source):
    request = InvestigationRequest("Compare capacities", "BTEST", 1, 3)
    selection = resolve_selection(request, sample_source)

    result = tools.compare_selection(selection, sample_source)

    assert result["selection"] == asdict(selection)
    assert result["comparison"]["capacity_change_Ah"] == pytest.approx(-0.5)
    assert result["comparison"]["capacity_change_percent"] == pytest.approx(-25.0)
    assert result["comparison"]["earlier"]["operation_index"] == 2
    assert result["comparison"]["later"]["operation_index"] == 6
    assert result["comparison"]["percentage_reference"] == "earlier selected discharge's reported capacity"
    assert result["comparison"]["comparability"]["status"] == "not_established"
    assert json.loads(json.dumps(result)) == result
    assert "measurements" not in result["comparison"]["earlier"]
    assert "measurements" not in result["comparison"]["later"]


def test_revised_selection_gets_recomputed_values_with_its_own_revision(sample_source):
    request = InvestigationRequest("Compare capacities", "BTEST", 1, 3)
    original_selection = resolve_selection(request, sample_source)
    original = tools.compare_selection(original_selection, sample_source)
    revised_selection = resolve_selection(replace(request, late=2), sample_source, revision=2)

    revised = tools.compare_selection(revised_selection, sample_source)

    assert original["selection"]["late"] == 3
    assert original["selection"]["revision"] == 1
    assert original["comparison"]["capacity_change_percent"] == pytest.approx(-25.0)
    assert revised["selection"]["late"] == 2
    assert revised["selection"]["late_operation_index"] == 4
    assert revised["selection"]["revision"] == 2
    assert revised["comparison"]["capacity_change_percent"] == pytest.approx(-10.0)
    assert revised["selection"]["source_sha256"] == original["selection"]["source_sha256"]


def test_changed_source_is_rejected_before_comparison(sample_source, monkeypatch):
    selection = resolve_selection(
        InvestigationRequest("Compare capacities", "BTEST", 1, 3), sample_source
    )
    with sample_source.open("ab") as stream:
        stream.write(b"changed")

    def unexpected_comparison(*args):
        pytest.fail("Comparison must not run against a different source")

    monkeypatch.setattr(tools, "compare_discharges", unexpected_comparison)
    with pytest.raises(ValueError, match="Source hash does not match"):
        tools.compare_selection(selection, sample_source)


def test_source_change_during_read_is_rejected_before_comparison(sample_source, monkeypatch):
    selection = resolve_selection(
        InvestigationRequest("Compare capacities", "BTEST", 1, 3), sample_source
    )
    original_reader = tools.read_discharge_records

    def read_then_change(source_path):
        result = original_reader(source_path)
        with source_path.open("ab") as stream:
            stream.write(b"changed")
        return result

    def unexpected_comparison(*args):
        pytest.fail("Comparison must not run after a source changes during reading")

    monkeypatch.setattr(tools, "read_discharge_records", read_then_change)
    monkeypatch.setattr(tools, "compare_discharges", unexpected_comparison)
    with pytest.raises(ValueError, match="Source changed while reading"):
        tools.compare_selection(selection, sample_source)


def test_wrong_source_name_is_rejected_before_reading(sample_source):
    selection = resolve_selection(
        InvestigationRequest("Compare capacities", "BTEST", 1, 3), sample_source
    )
    with pytest.raises(ValueError, match="does not match source BOTHER"):
        tools.compare_selection(selection, sample_source.with_name("BOTHER.mat"))


@pytest.mark.parametrize(
    "changes",
    [
        {"early_operation_index": 1},
        {"late_operation_index": 5},
        {"late": 2},
    ],
)
def test_inconsistent_operation_mapping_cannot_label_a_result(sample_source, changes):
    selection = resolve_selection(
        InvestigationRequest("Compare capacities", "BTEST", 1, 3), sample_source
    )
    inconsistent = replace(selection, **changes)
    with pytest.raises(ValueError, match="Operation indices do not match"):
        tools.compare_selection(inconsistent, sample_source)

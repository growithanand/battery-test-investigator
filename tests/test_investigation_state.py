"""Constructed sources verify request boundaries and selection provenance."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest
from scipy.io import savemat

from scripts import investigation_state as contracts


def write_source(tmp_path, time=(0.0, 1.0, 2.0)):
    operations = []
    for kind in ("charge", "discharge", "impedance", "discharge", "charge", "discharge"):
        operations.append({
            "type": kind,
            "ambient_temperature": 24,
            "data": {
                "Time": np.array(time),
                "Voltage_measured": np.array([4.1, 4.0, 3.9]),
                "Current_measured": np.array([-2.0, -2.0, -2.0]),
                "Temperature_measured": np.array([24.0, 24.2, 24.4]),
                "Capacity": 1.8,
            },
        })
    path = tmp_path / "BTEST.mat"
    savemat(path, {"BTEST": {"cycle": operations}})
    return path


def test_selection_keeps_source_hash_and_original_operation_positions(tmp_path):
    path = write_source(tmp_path)
    request = contracts.InvestigationRequest("Compare selected discharges", "BTEST", 2, 3)

    selection = contracts.resolve_selection(request, path)

    assert selection.battery_id == "BTEST"
    assert selection.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (selection.early, selection.late) == (2, 3)
    assert (selection.early_operation_index, selection.late_operation_index) == (4, 6)
    assert selection.revision == 1


@pytest.mark.parametrize(
    "fields,missing",
    [
        ({}, ("battery_id", "early", "late")),
        ({"battery_id": "BTEST", "early": 1}, ("late",)),
        ({"early": 1, "late": 3}, ("battery_id",)),
    ],
)
def test_incomplete_request_requires_clarification_before_reading_source(tmp_path, fields, missing):
    request = contracts.InvestigationRequest("Compare the tests", **fields)

    assert request.missing_fields() == missing
    with pytest.raises(ValueError, match="Request needs clarification"):
        contracts.resolve_selection(request, tmp_path / "not-present.mat")


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "1"])
@pytest.mark.parametrize("field", ["early", "late"])
def test_request_rejects_invalid_index_types_and_values(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        contracts.InvestigationRequest("Compare discharges", **{field: value})


@pytest.mark.parametrize("early,late", [(1, 1), (3, 1)])
def test_request_rejects_identical_or_reversed_selections(early, late):
    with pytest.raises(ValueError, match="early < late"):
        contracts.InvestigationRequest("Compare discharges", "BTEST", early, late)


@pytest.mark.parametrize(
    "fields",
    [{"question": ""}, {"question": " "}, {"battery_id": ""}, {"battery_id": " BTEST "}],
)
def test_request_rejects_blank_question_or_malformed_battery_id(fields):
    values = {"question": "Compare discharges", "battery_id": "BTEST"}
    values.update(fields)
    with pytest.raises(ValueError):
        contracts.InvestigationRequest(**values)


def test_missing_discharge_is_rejected_after_source_inspection(tmp_path):
    path = write_source(tmp_path)
    request = contracts.InvestigationRequest("Compare discharges", "BTEST", 1, 4)

    with pytest.raises(ValueError, match="early < late <= 3"):
        contracts.resolve_selection(request, path)


def test_battery_mismatch_is_rejected_before_reading_source(tmp_path):
    request = contracts.InvestigationRequest("Compare discharges", "BOTHER", 1, 2)
    with pytest.raises(ValueError, match="does not match source BTEST"):
        contracts.resolve_selection(request, tmp_path / "BTEST.mat")


def test_existing_reader_still_rejects_invalid_measurements(tmp_path):
    path = write_source(tmp_path, time=(0.0, 1.0, 1.0))
    request = contracts.InvestigationRequest("Compare discharges", "BTEST", 1, 3)
    with pytest.raises(ValueError, match="strictly increasing"):
        contracts.resolve_selection(request, path)


def test_source_change_during_resolution_is_rejected(tmp_path, monkeypatch):
    path = write_source(tmp_path)
    request = contracts.InvestigationRequest("Compare discharges", "BTEST", 1, 3)
    original_reader = contracts.read_discharge_records

    def read_then_change(source_path):
        result = original_reader(source_path)
        with source_path.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(contracts, "read_discharge_records", read_then_change)
    with pytest.raises(ValueError, match="Source changed"):
        contracts.resolve_selection(request, path)


@pytest.mark.parametrize("revision", [0, -1, True, 1.5])
def test_selection_revision_must_be_a_positive_integer(tmp_path, revision):
    request = contracts.InvestigationRequest("Compare discharges", "BTEST", 1, 3)
    with pytest.raises(ValueError, match="revision must be a positive integer"):
        contracts.resolve_selection(request, tmp_path / "BTEST.mat", revision=revision)


def test_revised_request_produces_a_distinct_selection_without_mutating_the_original(tmp_path):
    path = write_source(tmp_path)
    request = contracts.InvestigationRequest("Compare discharges", "BTEST", 1, 3)
    original = contracts.resolve_selection(request, path)

    revised_request = replace(request, late=2)
    revised = contracts.resolve_selection(revised_request, path, revision=2)

    assert request.late == 3
    assert original.late == 3
    assert original.late_operation_index == 6
    assert original.revision == 1
    assert revised.late == 2
    assert revised.late_operation_index == 4
    assert revised.revision == 2
    assert revised.source_sha256 == original.source_sha256

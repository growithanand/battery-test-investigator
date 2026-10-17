"""Constructed fixtures test data handling; these are not real battery results."""

import numpy as np
import pytest
from scipy.io import savemat

from scripts.inspect_sample import read_discharge_records


def operation(kind="discharge", time=None, voltage=None):
    return {
        "type": kind,
        "ambient_temperature": 24,
        "data": {
            "Time": np.array([0, 1, 2] if time is None else time),
            "Voltage_measured": np.array([4.1, 4.0, 3.9] if voltage is None else voltage),
            "Current_measured": np.array([-2.0, -2.0, -2.0]),
            "Temperature_measured": np.array([24.0, 24.2, 24.4]),
            "Capacity": 1.8,
        },
    }


def write_fixture(tmp_path, operations):
    path = tmp_path / "BTEST.mat"
    savemat(path, {"BTEST": {"cycle": operations}})
    return path


def test_original_positions_are_preserved_when_filtering_discharge(tmp_path):
    path = write_fixture(tmp_path, [operation("charge"), operation(), operation("impedance"), operation()])
    counts, records = read_discharge_records(path)
    assert counts == {"charge": 1, "discharge": 2, "impedance": 1}
    assert [row["operation_index"] for row in records] == [2, 4]
    assert [row["discharge_index"] for row in records] == [1, 2]


def test_repeated_elapsed_times_are_rejected(tmp_path):
    path = write_fixture(tmp_path, [operation(time=[0, 1, 1])])
    with pytest.raises(ValueError, match="strictly increasing"):
        read_discharge_records(path)


def test_misaligned_measurement_arrays_are_rejected(tmp_path):
    path = write_fixture(tmp_path, [operation(voltage=[4.1, 4.0])])
    with pytest.raises(ValueError, match="lengths must match"):
        read_discharge_records(path)


def test_nonfinite_measurements_are_rejected(tmp_path):
    path = write_fixture(tmp_path, [operation(voltage=[4.1, np.nan, 3.9])])
    with pytest.raises(ValueError, match="must be finite"):
        read_discharge_records(path)

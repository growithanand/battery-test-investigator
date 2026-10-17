"""Inspect recorded discharge capacity without inferring a failure mechanism."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ["operation_index", "discharge_index", "sample_count", "duration_s",
              "reported_capacity_Ah", "ambient_temperature_C", "median_current_A",
              "maximum_temperature_C"]


def read_discharge_records(path):
    """Keep original operation positions and number discharge operations separately."""
    cell = loadmat(path, simplify_cells=True)[path.stem]
    operations = cell["cycle"]
    if isinstance(operations, dict):
        operations = [operations]
    counts = dict(Counter(str(operation["type"]) for operation in operations))
    records = []
    for operation_index, operation in enumerate(operations, start=1):
        if operation["type"] != "discharge":
            continue
        data = operation["data"]
        arrays = {name: np.asarray(data[name], dtype=float).reshape(-1)
                  for name in ("Time", "Voltage_measured", "Current_measured", "Temperature_measured")}
        time = arrays["Time"]
        if time.size < 2 or any(array.size != time.size for array in arrays.values()):
            raise ValueError(f"Operation {operation_index}: measurement lengths must match and contain at least two samples")
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"Operation {operation_index}: measurements must be finite")
        if np.any(np.diff(time) <= 0):
            raise ValueError(f"Operation {operation_index}: elapsed times must be strictly increasing")
        capacity = float(data["Capacity"])
        ambient = float(operation["ambient_temperature"])
        if not np.isfinite(capacity) or capacity <= 0 or not np.isfinite(ambient):
            raise ValueError(f"Operation {operation_index}: capacity must be positive and metadata finite")
        records.append({
            "operation_index": operation_index,
            "discharge_index": len(records) + 1,
            "sample_count": int(time.size),
            "duration_s": float(time[-1] - time[0]),
            "reported_capacity_Ah": capacity,
            "ambient_temperature_C": ambient,
            "median_current_A": float(np.median(arrays["Current_measured"])),
            "maximum_temperature_C": float(np.max(arrays["Temperature_measured"])),
            "measurements": arrays,
        })
    if not records:
        raise ValueError("No discharge operations found")
    return counts, records


def save_plot(records, destination, cell_id):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), layout="constrained")
    axes[0].plot([r["discharge_index"] for r in records], [r["reported_capacity_Ah"] for r in records])
    axes[0].set(xlabel="Discharge operation number", ylabel="Reported capacity (Ah)",
                title=f"NASA {cell_id}: recorded discharge capacity")
    first = records[0]
    measurements = first["measurements"]
    axes[1].plot(measurements["Time"], measurements["Voltage_measured"], color="tab:orange")
    axes[1].set(xlabel="Elapsed time within operation (s)", ylabel="Voltage (V)",
                title=f"First discharge: original operation {first['operation_index']}")
    axes[2].plot(measurements["Time"], measurements["Temperature_measured"], color="tab:red")
    axes[2].set(xlabel="Elapsed time within operation (s)", ylabel="Battery temperature (°C)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/nasa/package-1/B0005.mat")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    counts, records = read_discharge_records(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_id = args.input.stem
    csv_path = args.output_dir / f"{cell_id}-discharge-summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "cell_id": cell_id,
        "source_filename": args.input.name,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "operation_counts": counts,
        "discharge_operations": len(records),
        "first_discharge": {key: records[0][key] for key in CSV_FIELDS},
        "last_discharge": {key: records[-1][key] for key in CSV_FIELDS},
        "interpretation": "Reported capacity and measurements only; no physical cause or end-of-life classification inferred.",
    }
    (args.output_dir / f"{cell_id}-inspection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_plot(records, args.output_dir / f"{cell_id}-inspection.png", cell_id)
    print(json.dumps(summary, indent=2))
    print(f"Inspection outputs: {args.output_dir}")


if __name__ == "__main__":
    main()

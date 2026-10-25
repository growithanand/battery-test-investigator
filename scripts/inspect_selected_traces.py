"""Inspect an explicit discharge pair without choosing an active-load boundary."""

import argparse
from pathlib import Path

from scripts.inspect_sample import ROOT
from scripts.investigation_state import InvestigationRequest, resolve_selection
from scripts.trace_inspection import write_trace_inspection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/nasa/package-1/B0005.mat")
    parser.add_argument("--early", type=int, required=True)
    parser.add_argument("--late", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/trace-inspections")
    args = parser.parse_args()
    try:
        request = InvestigationRequest("Inspect selected whole-record traces", args.input.stem, args.early, args.late)
        selection = resolve_selection(request, args.input)
        destination = write_trace_inspection(selection, args.input, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Trace inspection failed: {error}\n")
    print(f"Trace evidence: {destination / 'trace-inspection.json'}")
    print(f"Selected-pair plot: {destination / 'selected-traces.png'}")
    print("Whole recordings only; comparability remains not established.")


if __name__ == "__main__":
    main()

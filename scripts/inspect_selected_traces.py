"""Inspect whole-record traces or check the sampled load windows for B0005."""

import argparse
import json
from pathlib import Path

from scripts.inspect_sample import ROOT
from scripts.investigation_state import InvestigationRequest, resolve_selection
from scripts.trace_inspection import format_trace_extrema, inspect_b0005_active_load, write_trace_inspection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/nasa/package-1/B0005.mat")
    parser.add_argument("--early", type=int, required=True)
    parser.add_argument("--late", type=int, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output-dir", type=Path, default=ROOT / "outputs/trace-inspections")
    mode.add_argument("--active-load", action="store_true", help="Print the trial B0005 load windows as JSON")
    args = parser.parse_args()
    try:
        request = InvestigationRequest("Inspect selected whole-record traces", args.input.stem, args.early, args.late)
        selection = resolve_selection(request, args.input)
        if args.active_load:
            print(json.dumps(inspect_b0005_active_load(selection, args.input), indent=2, allow_nan=False))
            return
        destination = write_trace_inspection(selection, args.input, args.output_dir)
        extrema = format_trace_extrema(json.loads((destination / "trace-inspection.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Trace inspection failed: {error}\n")
    print(f"Trace evidence: {destination / 'trace-inspection.json'}")
    print(f"Selected-pair plot: {destination / 'selected-traces.png'}")
    print(extrema)
    print("Whole recordings only; comparability remains not established.")


if __name__ == "__main__":
    main()

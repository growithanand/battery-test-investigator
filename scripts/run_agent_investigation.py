"""Run a bounded local cited investigation with one model-selected trace follow-up."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import langsmith

from scripts.investigation_findings import FollowUpDecision, InvestigationFindings
from scripts.investigation_graph import WORKFLOW_VERSION, build_investigation_graph, investigation_prompts
from scripts.investigation_report import render_investigation_report
from scripts.investigation_resume import resume_investigation
from scripts.investigation_state import resolve_selection
from scripts.local_model import create_local_model
from scripts.model_tools import make_comparison_tool, make_protocol_tool, make_trace_tool
from scripts.protocol_evidence import CATALOG_PATH
from scripts.trace_inspection import ACTIVE_LOAD_ANALYSIS_VERSION, ANALYSIS_VERSION, format_trace_extrema


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION = "Compare reported capacities, document the protocol, and assess the conditions."
MODEL_SETTINGS = {"temperature": 0, "reasoning": False, "num_ctx": 8192,
                  "num_predict": 1024, "keep_alive": "2m"}


def _write_artifact(path: Path, content: str) -> None:
    """Publish a complete artifact via a temporary sibling."""
    pending = path.with_suffix(f"{path.suffix}.tmp")
    try:
        pending.write_text(content, encoding="utf-8", newline="\n")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact installed local tool-capable Ollama tag")
    question = parser.add_mutually_exclusive_group()
    question.add_argument("--question", help="Investigation question for a new run")
    question.add_argument("--resume", metavar="RUN_ID", help="Complete a saved clarification or revise its discharge pair")
    parser.add_argument("--battery-id", choices=["B0005"])
    parser.add_argument("--early", type=int)
    parser.add_argument("--late", type=int)
    parser.add_argument("--protocol-catalog", type=Path, default=CATALOG_PATH,
                        help="Trusted local source-catalog configuration, never a model argument")
    args = parser.parse_args()
    selection = {"battery_id": args.battery_id, "early": args.early, "late": args.late}
    resuming = args.resume is not None
    resume_kind = None
    validated_selection = None
    directory = ROOT / "outputs/agent-investigations"
    source = ROOT / "data/raw/nasa/package-1/B0005.mat"
    runtime = []

    def verify_source():
        manifest = json.loads((ROOT / "data/manifests/B0005.json").read_text(encoding="utf-8"))
        entry = next((item for item in manifest["files"] if item["archive_member"] == "B0005.mat"), None)
        if entry is None or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("B0005 source does not match the registered manifest")

    if resuming:
        try:
            completed, previous = resume_investigation(directory, args.resume, **selection)
            if completed.battery_id != source.stem:
                raise ValueError("Only the registered B0005 source can be resumed")
            verify_source()
            revision = 1 if previous is None else previous.revision + 1
            validated_selection = resolve_selection(completed, source, revision=revision)
            if previous is not None and validated_selection.source_sha256 != previous.source_sha256:
                raise ValueError("The registered source has changed since the saved investigation")
        except (OSError, ValueError) as error:
            parser.error(str(error))
        request = asdict(completed)
        resume_kind = "clarification" if previous is None else "selection_change"
    else:
        question_text = DEFAULT_QUESTION if args.question is None else args.question
        request = {"question": question_text, **selection}

    def load_model():
        # A missing selection reaches clarification without file reads or model preflight.
        if not runtime:
            verify_source()
            runtime.append(create_local_model(args.model, MODEL_SETTINGS))
        return runtime[0][0]

    limits = {"model_limit": 4, "tool_limit": 3}
    sources = {"B0005": source}
    graph = build_investigation_graph(
        load_model, make_comparison_tool(sources, validated_selection=validated_selection),
        make_protocol_tool(args.protocol_catalog),
        make_trace_tool(sources, include_active_load=True, validated_selection=validated_selection),
        resume_kind=resume_kind, **limits,
    )
    started = perf_counter()
    # Disable hosted tracing for the graph itself as well as the model/tool nodes.
    with langsmith.tracing_context(enabled=False):
        state = graph.invoke({"request": request}, {"recursion_limit": 12})
    # Never serialize raw AI messages or private reasoning content.
    result = {key: value for key, value in state.items() if key != "messages"}
    result.update({
        "run_id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "configuration": {
            "workflow_version": WORKFLOW_VERSION, "model_requested": args.model,
            "trace_analysis_version": ANALYSIS_VERSION,
            "active_load_analysis_version": ACTIVE_LOAD_ANALYSIS_VERSION,
            "model_used": runtime[0][1] if runtime else None, "limits": limits,
            "prompt_hashes": {name: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                              for name, prompt in investigation_prompts(resume_kind=resume_kind).items()},
            "follow_up_schema_sha256": hashlib.sha256(json.dumps(
                FollowUpDecision.model_json_schema(), sort_keys=True,
            ).encode("utf-8")).hexdigest(),
            "findings_schema_sha256": hashlib.sha256(json.dumps(
                InvestigationFindings.model_json_schema(),
                sort_keys=True,
            ).encode("utf-8")).hexdigest(),
            "packages": {name: version(name) for name in ("langchain", "langgraph", "langchain-ollama", "pydantic")},
        },
        "citation_check_scope": "Source ID, applicability and literal quote only; semantic support needs review.",
    })
    identity = {"request": result["request"], "configuration": result["configuration"],
                "selection": result["comparison"]["selection"] if result["comparison"] else None,
                "catalog_sha256": result["protocol"]["catalog_sha256"] if result["protocol"] else None}
    result["input_configuration_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    if resuming:
        result["resumed_from_run_id"] = args.resume
    output = directory / f"{result['run_id']}.json"
    directory.mkdir(parents=True, exist_ok=True)
    _write_artifact(output, json.dumps(result, indent=2) + "\n")
    print(f"Status: {result['status']} ({result['stop_reason']})")
    if resuming:
        print(f"Resumed from run: {args.resume}")
    if result["clarification"]:
        print(result["clarification"])
    if result["comparison"]:
        comparison = result["comparison"]["comparison"]
        print(f"Capacity change: {comparison['capacity_change_Ah']:.9f} Ah ({comparison['capacity_change_percent']:.6f}%)")
    if result["follow_up"]:
        print(f"Follow-up: {result['follow_up']['action']}: {result['follow_up']['reason']}")
    if result["trace_evidence"]:
        print(f"Selected-trace inspection executed: {result['trace_evidence']['evidence_id']}")
        print(format_trace_extrema(result["trace_evidence"]))
    if result["findings"]:
        for claim in result["findings"]["protocol_claims"]:
            print(f"Documented claim: {claim['statement']}\nQuote: {claim['quote']}\nEvidence: {claim['evidence_id']}")
        if result["findings"]["protocol_evidence_limitation"] is not None:
            print(f"Protocol limitation: {result['findings']['protocol_evidence_limitation']}")
        for limitation in result["findings"]["limitations"]:
            print(f"Limitation: {limitation}")
        print(f"Recommendation: {result['findings']['next_check']}")
    print(f"Local run record: {output}")
    report = output.with_suffix(".md")
    _write_artifact(report, render_investigation_report(result))
    print(f"Investigation report: {report}")
    if result["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

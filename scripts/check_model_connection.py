"""A bounded live tool round trip, not the complete investigation agent."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import langsmith

from scripts.investigation_state import InvestigationRequest
from scripts.local_model import create_local_model
from scripts.model_tools import ComparisonArguments, make_comparison_tool


ROOT = Path(__file__).resolve().parents[1]
MODEL_SETTINGS = {
    "temperature": 0,
    "reasoning": False,
    "num_ctx": 4096,
    "num_predict": 384,
    "keep_alive": "2m",
}
SYSTEM_PROMPT = (
    "You are checking a battery investigation tool connection. Use the comparison "
    "tool exactly once for the user's explicit selection. Never invent measurements. "
    "After the tool result, answer in at most four sentences with both supplied "
    "capacities, change in Ah and percent relative to the earlier capacity, and "
    "original operation indices. State that comparability is not established and "
    "no physical cause has been determined. No protocol documents are supplied."
)


def run_round_trip(model, comparison_tool, request: InvestigationRequest) -> dict:
    """Allow two model calls and one validated tool call; never retry silently."""
    if request.missing_fields():
        raise ValueError("An explicit battery, early discharge, and late discharge are required")
    expected = {"battery_id": request.battery_id, "early": request.early, "late": request.late}
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"{request.question}\nExplicit selection: {json.dumps(expected)}"
        )),
    ]
    started = perf_counter()
    # Keep this local even if hosted tracing is enabled in the parent environment.
    with langsmith.tracing_context(enabled=False):
        decision = model.bind_tools([comparison_tool]).invoke(messages)
        if decision.invalid_tool_calls or len(decision.tool_calls) != 1:
            raise ValueError("Expected exactly one valid comparison tool call")
        call = decision.tool_calls[0]
        if call["name"] != comparison_tool.name:
            raise ValueError("Model requested an unregistered tool")
        arguments = ComparisonArguments.model_validate(call["args"]).model_dump()
        if arguments != expected:
            raise ValueError("Model tool arguments differ from the explicit user selection")

        result = comparison_tool.invoke(arguments)
        messages.extend([
            decision,
            ToolMessage(content=json.dumps(result), tool_call_id=call["id"]),
        ])
        # Do not bind tools on this final call: there is no agent loop or extra budget.
        answer = model.invoke(messages)
        if answer.tool_calls or answer.invalid_tool_calls or not answer.text.strip():
            raise ValueError("Expected a non-empty final answer without further tool calls")
        if answer.response_metadata.get("done_reason") == "length":
            raise ValueError("Final answer reached the output-token limit")

    return {
        "status": "round_trip_completed",
        "scope": "connection check; generated answer is not an evaluated investigation",
        "request": {"question": request.question, **expected},
        "tool_call": {"name": call["name"], "args": arguments, "id": call["id"]},
        "tool_result": result,
        "answer": answer.text,
        "model_calls": 2,
        "tool_calls": 1,
        "elapsed_seconds": perf_counter() - started,
        "usage": [decision.usage_metadata, answer.usage_metadata],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact already-installed local Ollama tag")
    parser.add_argument("--early", type=int, default=1)
    parser.add_argument("--late", type=int, default=168)
    args = parser.parse_args()
    request = InvestigationRequest("Compare reported discharge capacities.", "B0005", args.early, args.late)

    # Read only the registered manifest path, never a path supplied by the model.
    manifest = json.loads((ROOT / "data/manifests/B0005.json").read_text(encoding="utf-8"))
    source = ROOT / "data/raw/nasa/package-1/B0005.mat"
    entry = next(item for item in manifest["files"] if item["path"] == source.relative_to(ROOT).as_posix())
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("B0005 source does not match the registered manifest")

    model, model_metadata = create_local_model(args.model, MODEL_SETTINGS)
    comparison_tool = make_comparison_tool({"B0005": source})
    trace = run_round_trip(model, comparison_tool, request)
    trace.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        **model_metadata,
        "packages": {name: version(name) for name in (
            "langchain", "langchain-core", "langchain-ollama", "langgraph", "ollama", "pydantic"
        )},
    })
    output = ROOT / "outputs/model-connection" / f"round-trip-{uuid4().hex}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(trace["answer"])
    print(f"\nLocal trace: {output}")


if __name__ == "__main__":
    main()

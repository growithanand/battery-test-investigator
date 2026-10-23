"""Synthetic tools/messages check boundaries; live model evidence is separate."""

import json

import numpy as np
import pytest
from scipy.io import savemat

pytest.importorskip("langchain_ollama", reason="Install requirements-agent-lock.txt for model-layer tests")

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from scripts.check_model_connection import run_round_trip
from scripts.investigation_state import InvestigationRequest
from scripts.model_tools import ComparisonArguments, make_comparison_tool


@pytest.fixture
def comparison_tool(tmp_path):
    operations = []
    for capacity in (2.0, 1.5):
        operations.append({
            "type": "discharge", "ambient_temperature": 24,
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
    return make_comparison_tool({"BTEST": path})


class StubModel:
    """Scripted test double, not evidence of a live tool decision."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def tool_decision(**changes):
    call = {"name": "compare_discharge_capacity", "args": {
        "battery_id": "BTEST", "early": 1, "late": 2,
    }, "id": "test-call"}
    call.update(changes)
    return AIMessage(content="", tool_calls=[call])


def test_tool_returns_real_calculation_not_model_supplied_numbers(comparison_tool):
    result = comparison_tool.invoke({"battery_id": "BTEST", "early": 1, "late": 2})
    assert result["comparison"]["capacity_change_percent"] == pytest.approx(-25.0)
    assert result["selection"]["late_operation_index"] == 2
    assert len(result["selection"]["source_sha256"]) == 64
    assert result["comparison"]["comparability"]["status"] == "not_established"
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("changes", [
    {"early": True}, {"early": "1"}, {"late": 2.0}, {"late": 0},
    {"source_path": "untrusted.mat"}, {"revision": 9}, {"capacity_change_percent": -50},
])
def test_schema_rejects_coercion_paths_and_claimed_results(changes):
    arguments = {"battery_id": "BTEST", "early": 1, "late": 2, **changes}
    with pytest.raises(ValidationError):
        make_comparison_tool({}).invoke(arguments)


def test_schema_exposes_only_selection_fields():
    assert set(ComparisonArguments.model_fields) == {"battery_id", "early", "late"}


def test_unknown_battery_is_not_treated_as_a_path():
    with pytest.raises(ValueError, match="not registered"):
        make_comparison_tool({}).invoke({"battery_id": "../BTEST.mat", "early": 1, "late": 2})


def test_reversed_indices_fail_before_source_read():
    with pytest.raises(ValueError, match="early < late"):
        make_comparison_tool({"BTEST": "missing.mat"}).invoke({
            "battery_id": "BTEST", "early": 2, "late": 1,
        })


def test_round_trip_links_call_id_and_result_then_stops(comparison_tool):
    model = StubModel(tool_decision(), AIMessage(content="Synthetic final response"))
    trace = run_round_trip(model, comparison_tool, InvestigationRequest("Compare", "BTEST", 1, 2))
    assert len(model.calls) == trace["model_calls"] == 2
    assert trace["tool_calls"] == 1
    tool_message = model.calls[1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "test-call"
    assert json.loads(tool_message.content) == trace["tool_result"]
    assert trace["tool_result"]["comparison"]["capacity_change_percent"] == pytest.approx(-25.0)


@pytest.mark.parametrize("decision, error", [
    (AIMessage(content="I guessed the answer"), "exactly one"),
    (tool_decision(name="read_arbitrary_file"), "unregistered tool"),
    (tool_decision(args={"battery_id": "BTEST", "early": 1, "late": 3}), "explicit user selection"),
    (AIMessage(content="", tool_calls=tool_decision().tool_calls * 2), "exactly one"),
])
def test_bad_decision_stops_before_any_tool_execution(decision, error, monkeypatch):
    tool = make_comparison_tool({})
    monkeypatch.setattr(type(tool), "invoke", lambda *a, **kw: pytest.fail("Tool must not run"))
    model = StubModel(decision)
    with pytest.raises(ValueError, match=error):
        run_round_trip(model, tool, InvestigationRequest("Compare", "BTEST", 1, 2))
    assert len(model.calls) == 1


def test_incomplete_request_does_not_call_model():
    model = StubModel()
    with pytest.raises(ValueError, match="explicit battery"):
        run_round_trip(model, make_comparison_tool({}), InvestigationRequest("Compare"))
    assert not model.calls


@pytest.mark.parametrize("answer", [
    AIMessage(content=""), tool_decision(),
    AIMessage(content="Truncated", response_metadata={"done_reason": "length"}),
])
def test_no_retry_or_further_tool_after_bad_final_answer(comparison_tool, answer):
    model = StubModel(tool_decision(), answer)
    with pytest.raises(ValueError):
        run_round_trip(model, comparison_tool, InvestigationRequest("Compare", "BTEST", 1, 2))
    assert len(model.calls) == 2

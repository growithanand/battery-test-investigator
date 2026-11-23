"""Mock Gemini responses exercise the adapter without network access or real keys."""

from copy import deepcopy
import json

import httpx
import pytest

pytest.importorskip("langchain_core", reason="Install the optional agent dependency layer")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from scripts import gemini_model
from scripts.investigation_findings import FollowUpDecision
from scripts.model_tools import make_comparison_tool, make_protocol_tool
from scripts.model_usage import summarize_model_usage


MODEL = "gemini-test-model"
CANARY = "synthetic-private-value"
DECISION = {"action": "stop_with_partial", "question_scope": "reported_capacity", "reason": "The requested capacity calculation is complete."}


def response_body(parts=None, **changes):
    return {"modelVersion": MODEL,
            "candidates": [{"content": {"role": "model", "parts": parts or [{"text": json.dumps(DECISION)}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5, "totalTokenCount": 125},
            **changes}


def model_with_response(body, status=200):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    return gemini_model.GeminiChat(MODEL, client), requests


@pytest.mark.parametrize("provider_ids", [True, False])
def test_tool_round_trip_preserves_original_parts_signatures_and_reply_ids(provider_ids):
    parts = [{"functionCall": {"name": name, "args": args}, "thoughtSignature": signature}
             for name, args, signature in (
                 ("compare_discharge_capacity", {"battery_id": "B0005", "early": 1, "late": 2}, "opaque-first"),
                 ("get_battery_protocol", {"battery_id": "B0005"}, "opaque-second"))]
    if provider_ids:
        for index, part in enumerate(parts):
            part["functionCall"]["id"] = f"provider-{index}"
    original = deepcopy(parts)
    model, requests = model_with_response(response_body(parts))
    messages = [SystemMessage(content="Use only registered tools."), HumanMessage(content="Compare this pair.")]
    try:
        tools = [make_comparison_tool({}), make_protocol_tool()]
        decision = model.bind_tools(tools).invoke(messages)
        assert len({call["id"] for call in decision.tool_calls}) == 2
        replies = [ToolMessage(content=json.dumps({"checked": call["name"]}), tool_call_id=call["id"])
                   for call in decision.tool_calls]
        model.bind_tools(tools).invoke([*messages, decision, *replies])
        payload = json.loads(requests[1].content)
        assert payload["contents"][-2] == {"role": "model", "parts": original}
        responses = [part["functionResponse"] for part in payload["contents"][-1]["parts"]]
        assert [reply["name"] for reply in responses] == [part["functionCall"]["name"] for part in original]
        if provider_ids:
            assert [reply["id"] for reply in responses] == ["provider-0", "provider-1"]
        else:
            assert all("id" not in reply for reply in responses)
            assert not {call["id"] for call in decision.tool_calls}.intersection(
                call["id"] for call in model.bind_tools(tools).invoke(messages).tool_calls)
        assert payload["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["additionalProperties"] is False
        assert "opaque-first" not in json.dumps(model.calls)
    finally:
        model.close()


def test_structured_findings_are_validated_and_thought_text_is_not_exposed():
    parts = [{"thought": True, "text": CANARY, "thoughtSignature": "opaque-signature"}, {"text": json.dumps(DECISION)}]
    model, requests = model_with_response(response_body(parts))
    try:
        result = model.with_structured_output(FollowUpDecision, method="json_schema", include_raw=True).invoke([HumanMessage(content="Choose.")])
        assert result["parsed"].model_dump() == DECISION
        assert CANARY not in result["raw"].content
        assert result["raw"].usage_metadata == {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}
        payload = json.loads(requests[0].content)
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        assert payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
        assert payload["generationConfig"]["maxOutputTokens"] == 1024
        assert "tools" not in payload
        assert CANARY not in json.dumps(model.calls)
    finally:
        model.close()


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_http_failure_does_not_echo_provider_body_or_retry(status):
    model, requests = model_with_response({"error": {"message": CANARY}}, status)
    try:
        with pytest.raises(gemini_model.GeminiModelError, match=str(status)) as error:
            model.invoke([HumanMessage(content="Compare.")])
        assert CANARY not in str(error.value)
        assert len(requests) == len(model.calls) == 1
        assert model.calls[0]["http_status"] == status and model.calls[0]["usage"] is None
    finally:
        model.close()


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError])
def test_transport_failure_is_recorded_without_sensitive_exception_details(failure):
    requests = []

    def respond(request):
        requests.append(request)
        raise failure(CANARY, request=request)

    model = gemini_model.GeminiChat(MODEL, httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(gemini_model.GeminiModelError) as error:
            model.invoke([HumanMessage(content="Compare.")])
        assert CANARY not in str(error.value) and CANARY not in json.dumps(model.calls)
        assert len(requests) == 1 and model.calls[0]["seconds"] >= 0
        assert model.calls[0]["error"] in ("timeout", "connection_error")
    finally:
        model.close()


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY", None])
def test_incomplete_output_is_rejected_but_reported_usage_is_retained(finish):
    body = response_body()
    body["candidates"][0]["finishReason"] = finish
    model, _ = model_with_response(body)
    try:
        with pytest.raises(gemini_model.GeminiModelError, match="blocked or incomplete"):
            model.invoke([HumanMessage(content="Compare.")])
        usage = summarize_model_usage({"model_attempts": 1}, model.calls)
        assert usage["complete"] and usage["total_tokens"] == 125
    finally:
        model.close()


@pytest.mark.parametrize("body", [[], {}, {"candidates": []}, {"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": []}}]},
                                  response_body([{"functionCall": {"name": "test", "args": CANARY}}]),
                                  response_body([{"functionCall": {"name": "test", "args": {}, "id": 42}}]),
                                  response_body([{"text": {"private": CANARY}}])])
def test_malformed_response_errors_are_safe(body):
    model, _ = model_with_response(body)
    try:
        with pytest.raises(gemini_model.GeminiModelError) as error:
            model.invoke([HumanMessage(content="Compare.")])
        assert CANARY not in str(error.value)
    finally:
        model.close()


@pytest.mark.parametrize("text", ["not-json " + CANARY, json.dumps({**DECISION, "unexpected": CANARY})])
def test_invalid_structured_output_hides_validation_inputs(text):
    model, _ = model_with_response(response_body([{"text": text}]))
    try:
        with pytest.raises(gemini_model.GeminiModelError, match="invalid structured findings") as error:
            model.with_structured_output(FollowUpDecision, method="json_schema", include_raw=True).invoke([HumanMessage(content="Choose.")])
        assert CANARY not in str(error.value)
    finally:
        model.close()


@pytest.mark.parametrize("usage", [None, {}, {"promptTokenCount": 100},
                                   {"promptTokenCount": -1, "totalTokenCount": 0},
                                   {"promptTokenCount": True, "totalTokenCount": 1},
                                   {"promptTokenCount": 100, "candidatesTokenCount": 20, "totalTokenCount": 999}])
def test_missing_or_inconsistent_usage_is_unknown(usage):
    model, _ = model_with_response(response_body(usageMetadata=usage))
    try:
        result = model.invoke([HumanMessage(content="Choose.")])
        assert result.usage_metadata is None
        assert not summarize_model_usage({"model_attempts": 1}, model.calls)["complete"]
    finally:
        model.close()


def test_usage_counts_custom_tool_prompt_tokens_and_does_not_add_cached_tokens_twice():
    body = {"usageMetadata": {"promptTokenCount": 100, "toolUsePromptTokenCount": 10, "candidatesTokenCount": 20,
                              "thoughtsTokenCount": 5, "cachedContentTokenCount": 40, "totalTokenCount": 135}}
    assert gemini_model.read_usage(body) == {"input_tokens": 110, "output_tokens": 25, "total_tokens": 135}


def test_factory_uses_key_header_fixed_host_and_safe_metadata(monkeypatch):
    created = []
    real_client = httpx.Client

    def client(**kwargs):
        assert kwargs == {"headers": {"x-goog-api-key": CANARY}, "timeout": 40.0, "trust_env": False, "follow_redirects": False}
        created.append(real_client(**kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response_body()))))
        return created[-1]

    monkeypatch.setattr(gemini_model, "read_gemini_key", lambda: CANARY)
    monkeypatch.setattr(gemini_model.httpx, "Client", client)
    model, metadata = gemini_model.create_gemini_model(MODEL)
    assert not created[0].is_closed and CANARY not in json.dumps(metadata)
    model.close()
    assert created[0].is_closed


@pytest.mark.parametrize("tag", ["https://remote.invalid/model", "gemini-test?key=value", "gemini-test/other", "models/gemini-test", "local-tag:1"])
def test_invalid_model_ids_are_rejected_before_key_or_network_access(monkeypatch, tag):
    monkeypatch.setattr(gemini_model, "read_gemini_key", lambda: pytest.fail("Key must not be read"))
    with pytest.raises(gemini_model.GeminiModelError, match="model ID"):
        gemini_model.create_gemini_model(tag)


def test_unreadable_key_file_has_a_safe_error(monkeypatch):
    def unreadable():
        raise OSError(CANARY)

    monkeypatch.setattr(gemini_model, "read_gemini_key", unreadable)
    with pytest.raises(gemini_model.GeminiModelError, match="Could not read .env") as error:
        gemini_model.create_gemini_model(MODEL)
    assert CANARY not in str(error.value)


def test_schema_conversion_preserves_references_without_mutating_original():
    original = {"$defs": {"Scope": {"const": "reported_capacity", "type": "string"}}, "properties": {"scope": {"$ref": "#/$defs/Scope"}}}
    before = deepcopy(original)
    result = gemini_model.schema_for_api(original)
    assert original == before
    assert result["$defs"]["Scope"] == {"enum": ["reported_capacity"], "type": "string"}
    assert result["properties"] == original["properties"]


def test_local_usage_summary_marks_a_failed_attempt_as_incomplete():
    usage = {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}
    state = {"model_attempts": 2, "trace": [{"event": "tools_requested", "usage": usage}, {"event": "follow_up_error"}]}
    summary = summarize_model_usage(state)
    assert summary == {"complete": False, "requests_with_usage": 1, "model_attempts": 2, **usage}
    assert summarize_model_usage({"model_attempts": 0, "trace": []})["complete"]
